from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse
from tqdm import tqdm

import numpy as np
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from torchvision import transforms
from ImageDataLoader import SimpleImageLoader
from models import contrastive_model
from loss import ContrastiveLoss
import nsml
from nsml import DATASET_PATH, IS_ON_NSML

NUM_CLASSES = 265
# code from baseline
def top_n_accuracy_score(y_true, y_prob, n=5, normalize=True):
    num_obs, num_labels = y_prob.shape
    idx = num_labels - n - 1
    counter = 0
    argsorted = np.argsort(y_prob, axis=1)
    for i in range(num_obs):
        if y_true[i] in argsorted[i, idx+1:]:
            counter += 1
    if normalize:
        return counter * 1.0 / num_obs
    else:
        return counter

# code from baseline
def split_ids(path, ratio):
    with open(path) as f:
        ids_l = []
        ids_u = []
        for i, line in enumerate(f.readlines()):
            if i == 0 or line == '' or line == '\n':
                continue
            line = line.replace('\n', '').split('\t')
            if int(line[1]) >= 0:
                ids_l.append(int(line[0]))
            else:
                ids_u.append(int(line[0]))

    ids_l = np.array(ids_l)
    ids_u = np.array(ids_u)

    perm = np.random.permutation(np.arange(len(ids_l)))
    cut = int(ratio*len(ids_l))
    train_ids = ids_l[perm][cut:]
    val_ids = ids_l[perm][:cut]

    return train_ids, val_ids, ids_u

### NSML functions
def _infer(model, root_path, test_loader=None):
    if test_loader is None:
        test_loader = torch.utils.data.DataLoader(
            SimpleImageLoader(root_path, 'test',
                               transform=transforms.Compose([
                                   transforms.Resize(opts.imResize),
                                   transforms.CenterCrop(opts.imsize),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                               ])), batch_size=opts.batchsize_label, shuffle=False, num_workers=4, pin_memory=True)
        print('loaded {} test images'.format(len(test_loader.dataset)))

    outputs = []
    s_t = time.time()
    for idx, image in enumerate(test_loader):
        if torch.cuda.is_available():
            image = image.cuda()
        _, probs = model(image)
        output = torch.argmax(probs, dim=1)
        output = output.detach().cpu().numpy()
        outputs.append(output)

    outputs = np.concatenate(outputs)
    return outputs

def bind_nsml(model):
    def save(dir_name, *args, **kwargs):
        os.makedirs(dir_name, exist_ok=True)
        state = model.state_dict()
        torch.save(state, os.path.join(dir_name, 'model.pt'))
        print('saved')

    def load(dir_name, *args, **kwargs):
        state = torch.load(os.path.join(dir_name, 'model.pt'))
        state = {k.replace('module.', ''): v for k, v in state.items()}
        model.load_state_dict(state)
        print('loaded')

    def infer(root_path):
        return _infer(model, root_path)

    nsml.bind(save=save, load=load, infer=infer)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Options
######################################################################
parser = argparse.ArgumentParser(description='SimCLR only training & combined training')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N', help='number of start epoch (default: 1)')
parser.add_argument('--steps_per_epoch', type=int, default=-1, metavar='N', help='number of step during one epoch (default: -1)')
parser.add_argument('--warmup_epoch', type=int, default=10, metavar='N', help='number of epochs for warm-up')

# basic settings
parser.add_argument('--name',default='c_model', type=str, help='output model name')
parser.add_argument('--gpu_ids',default='0,1', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--batchsize_unlabel', default=128, type=int, help='batchsize for unlabeled dataset')
parser.add_argument('--batchsize_label', default=4, type=int, help='batchsize for labeled dataset')
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--num_worker', type=int, default=2, help='num_worker')

# basic hyper-parameters
parser.add_argument('--momentum', type=float, default=0.9, metavar='LR', help='momentum value for optimizer')
parser.add_argument('--lr', type=float, default=1.0, metavar='LR', help='initial learning rate')
parser.add_argument('--combined_lr', type=float, default=0.00001, metavar='LR', help='initial learning rate for combined training')
parser.add_argument('--smoothing', type=float, default=0.01, help='smoothing value for label smoothing')
parser.add_argument('--imResize', default=256, type=int, help='')
parser.add_argument('--imsize', default=224, type=int, help='')

# arguments for logging and backup
parser.add_argument('--log_interval', type=int, default=10, metavar='N', help='logging training status')
parser.add_argument('--save_epoch', type=int, default=20, help='saving epoch interval')

# hyper-params only simclr training
#parser.add_argument('--use_pretrained', type=bool, default=False, help='if True, use pretrained model with simclr loss by 300epochs')
#parser.add_argument('--epochs', type=int, default=300, metavar='N', help='number of epochs to train (default: 400)')
#parser.add_argument('--consistency_epoch', type=int, default=300, metavar='N', help='number of epochs to consistency training (default : 300)')

# hyper-params combined training after simclr training
parser.add_argument('--use_pretrained', type=bool, default=True, help='if True, use pretrained model with simclr loss by 300epochs')
parser.add_argument('--ckpt', type=str, default='c_model_e299', help='pretrained ckpt name')
parser.add_argument('--session', type=str, default='kaist006/fashion_dataset/51', help='session name which contain pretrained ckpt')
parser.add_argument('--epochs', type=int, default=400, metavar='N', help='number of epochs to train (default: 400)')
parser.add_argument('--consistency_epoch', type=int, default=300, metavar='N', help='number of epochs to consistency training (default : 300)')

# hyper-parameters for sim-CLR
parser.add_argument('--out_dim', type=int, default=128, metavar='N', help='feature dimension for sim_CLR')
parser.add_argument('--temperature', type=float, default=1.0, metavar='N', help='Temperature for sharpening')

### DO NOT MODIFY THIS BLOCK ###
# arguments for nsml 
parser.add_argument('--pause', type=int, default=0)
parser.add_argument('--mode', type=str, default='train')
################################

def main():
    global opts, global_step
    opts = parser.parse_args()
    opts.cuda = 0
    global_step = 0
    print(opts)

    seed = opts.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        opts.cuda = 1
        print("Currently using GPU {}".format(opts.gpu_ids))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    # Set model
    model = contrastive_model(NUM_CLASSES, opts.out_dim)

    if use_gpu:
        model.cuda()
    
    ### DO NOT MODIFY THIS BLOCK ###
    if IS_ON_NSML:
        bind_nsml(model)
        if opts.pause:
            nsml.paused(scope=locals())
    ################################

    if opts.mode == 'train':
        # set multi-gpu
        if len(opts.gpu_ids.split(',')) > 1:
            model = nn.DataParallel(model)
        model.train()

        # Set dataloader
        train_ids, val_ids, unl_ids = split_ids(os.path.join(DATASET_PATH, 'train/train_label'), 0.2)
        print('found {} train, {} validation and {} unlabeled images'.format(len(train_ids), len(val_ids), len(unl_ids)))

        # Set transforms for train
        train_transforms = transforms.Compose([
            transforms.Resize(opts.imResize),
            transforms.RandomResizedCrop(opts.imsize, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        eval_transforms = transforms.Compose([
            transforms.Resize(opts.imResize),
            transforms.CenterCrop(opts.imsize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        train_loader = torch.utils.data.DataLoader(
            SimpleImageLoader(DATASET_PATH, 'train', train_ids,
                                transform=train_transforms),
                                batch_size=opts.batchsize_label, shuffle=True, num_workers=opts.num_worker, pin_memory=True, drop_last=True)
        print('train_loader done')

        unlabel_loader = torch.utils.data.DataLoader(
            SimpleImageLoader(DATASET_PATH, 'unlabel', unl_ids,
                                transform=train_transforms),
                                batch_size=opts.batchsize_unlabel, shuffle=True, num_workers=opts.num_worker, pin_memory=True, drop_last=True)
        print('unlabel_loader done')    

        validation_loader = torch.utils.data.DataLoader(
            SimpleImageLoader(DATASET_PATH, 'val', val_ids,
                                transform=eval_transforms),
                                batch_size=opts.batchsize_label, shuffle=False, num_workers=opts.num_worker, pin_memory=True, drop_last=False)
        print('validation_loader done')

        # Set optimizer
        if opts.use_pretrained:
            optimizer = optim.SGD(model.parameters(), lr=opts.combined_lr, momentum=opts.momentum, weight_decay=1e-6)
        else:
            optimizer = optim.SGD(model.parameters(), lr=opts.lr, momentum=opts.momentum, weight_decay=1e-6)
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(unlabel_loader), eta_min=0, last_epoch=-1)
        criterion = ContrastiveLoss(opts.temperature, opts.consistency_epoch, opts.batchsize_label, opts.batchsize_unlabel, NUM_CLASSES, opts.smoothing)
        opts.steps_per_epoch = len(unlabel_loader)

        # Train and Validation
        best_acc = 0
        # Load model from other session
        if opts.use_pretrained:
            nsml.load(checkpoint=opts.ckpt, session=opts.session)
            nsml.save('saved')

        for epoch in range(opts.start_epoch, opts.epochs + 1):
            if opts.use_pretrained:
                if epoch <= opts.consistency_epoch:
                    continue

                if epoch > opts.consistency_epoch and epoch <= opts.consistency_epoch + opts.warmup_epoch:
                    for g in optimizer.param_groups:
                        g['lr'] = opts.combined_lr * (epoch - opts.consistency_epoch) / opts.warmup_epoch
                else:
                    scheduler.step()
            else:
                if epoch <= opts.warmup_epoch : # warm up
                    for g in optimizer.param_groups:
                        g['lr'] = opts.lr * epoch / opts.warmup_epoch
                else:
                    scheduler.step()

            loss, avg_top1, avg_top5 = train(opts, train_loader, unlabel_loader, model, criterion, optimizer, epoch, use_gpu)
            print('epoch {:03d}/{:03d} finished, loss: {:.3f}, avg_top1: {:.3f}%, avg_top5: {:.3f}%'.format(epoch, opts.epochs, loss, avg_top1, avg_top5))
            
            acc_top1, acc_top5 = validation(opts, validation_loader, model, epoch, use_gpu)
            is_best = acc_top1 > best_acc
            if is_best:
                best_acc = acc_top1
                print('model achieved the best accuracy ({:.3f}%) - saving best checkpoint...'.format(best_acc))
                if IS_ON_NSML:
                    nsml.save(opts.name + '_best')
                else:
                    torch.save(model.state_dict(), os.path.join('runs', opts.name + '_best'))
            if (epoch + 1) % opts.save_epoch == 0:
                if IS_ON_NSML:
                    nsml.save(opts.name + '_e{}'.format(epoch))
                else:
                    torch.save(model.state_dict(), os.path.join('runs', opts.name + '_e{}'.format(epoch)))

def train(opts, train_loader, unlabel_loader, model, criterion, optimizer, epoch, use_gpu):
    global global_step

    losses = AverageMeter()
    losses_curr = AverageMeter()
    acc_top1 = AverageMeter()
    acc_top5 = AverageMeter()
    
    model.train()

    out = False
    local_step = 0
    while not out:
        labeled_train_iter = iter(train_loader)
        unlabeled_train_iter = iter(unlabel_loader)
        for batch_idx in range(len(unlabel_loader)):
            optimizer.zero_grad()
            try:
                data = labeled_train_iter.next()
                inputs_x1, inputs_x2, targets_x = data
            except:
                labeled_train_iter = iter(train_loader)       
                data = labeled_train_iter.next()
                inputs_x1, inputs_x2, targets_x = data
            try:
                data = unlabeled_train_iter.next()
                inputs_u1, inputs_u2 = data
            except:
                unlabeled_train_iter = iter(unlabel_loader)       
                data = unlabeled_train_iter.next()
                inputs_u1, inputs_u2 = data

            if use_gpu:
                inputs_x1, inputs_x2, targets_x = inputs_x1.cuda(), inputs_x2.cuda(), targets_x.cuda()
                inputs_u1, inputs_u2 = inputs_u1.cuda(), inputs_u2.cuda()
            
            embed_x1, pred_x1 = model(inputs_x1)
            embed_x2, pred_x2 = model(inputs_x2)
            embed_u1, _ = model(inputs_u1)
            embed_u2, _ = model(inputs_u2)

            loss = criterion(embed_x1, embed_x2, pred_x1, pred_x2, targets_x, embed_u1, embed_u2, epoch)
            losses.update(loss.item(), inputs_x1.size(0) + inputs_u1.size(0))
            losses_curr.update(loss.item(), inputs_x1.size(0) + inputs_u1.size(0))
            loss.backward()
            optimizer.step()

            if IS_ON_NSML and global_step % opts.log_interval == 0:
                nsml.report(step=global_step, loss=losses_curr.avg)
                losses_curr.reset()

            if epoch <= opts.consistency_epoch:
                acc_top1_1 = -1
                acc_top1_2 = -1
                acc_top5_1 = -1
                acc_top5_2 = -1
            else:
                acc_top1_1 = top_n_accuracy_score(targets_x.data.cpu().numpy(), pred_x1.data.cpu().numpy(), n=1)*100
                acc_top1_2 = top_n_accuracy_score(targets_x.data.cpu().numpy(), pred_x2.data.cpu().numpy(), n=1)*100
                acc_top5_1 = top_n_accuracy_score(targets_x.data.cpu().numpy(), pred_x1.data.cpu().numpy(), n=5)*100
                acc_top5_2 = top_n_accuracy_score(targets_x.data.cpu().numpy(), pred_x2.data.cpu().numpy(), n=5)*100
            acc_top1.update(acc_top1_1, inputs_x1.size(0))
            acc_top1.update(acc_top1_2, inputs_x2.size(0))
            acc_top5.update(acc_top5_1, inputs_x1.size(0))
            acc_top5.update(acc_top5_2, inputs_x2.size(0))

            local_step += 1
            global_step += 1

            if local_step >= opts.steps_per_epoch:
                out = True
                break

    return losses.avg, acc_top1.avg, acc_top5.avg

def validation(opts, validation_loader, model, epoch, use_gpu):
    if epoch <= opts.consistency_epoch:
        return -1, -1
    
    model.eval()
    avg_top1 = 0.0
    avg_top5 = 0.0
    nCnt = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(validation_loader):
            inputs, labels = data
            if use_gpu:
                inputs = inputs.cuda()
            nCnt += 1
            embed_fea, preds = model(inputs)

            acc_top1 = top_n_accuracy_score(labels.numpy(), preds.data.cpu().numpy(), n=1)*100
            acc_top5 = top_n_accuracy_score(labels.numpy(), preds.data.cpu().numpy(), n=5)*100
            avg_top1 += acc_top1
            avg_top5 += acc_top5
        
        avg_top1 = float(avg_top1/nCnt)
        avg_top5 = float(avg_top5/nCnt)

    if IS_ON_NSML:
        nsml.report(step=epoch, avg_top1=avg_top1, avg_top5=avg_top5)

    return avg_top1, avg_top5

if __name__ == '__main__':
    main()
            









