# CS_492_CV
CS_492 : semi-supervised learning for naver fashion dataset

In this repository, There are two folders: project1, project1_arc

## project1
    Project1 was created for two purposes. The first is to train the model for 300 epochs using only sim-clr Loss function. This pre-trained model was used in subsequent experiments. The second is The second is to train the previously pre-trained model with sim-clr loss and standard cross entropy loss. When calculating cross entropy loss, we applied label smoothing.

### ImageDataLoader.py
    When training time, we need to apply two differnet transforms on same images for labeled and unlabeled data and get its label of labeled data. When valid and test time, we need single image and its label. To do this, __getitem__ function will return images and label depend on data type. 

### loss.py
    ContrastiveLoss class calculates combined loss value during training time. When we train the model with only sim-clr loss, ContrastiveLoss class will calculate only NTXentLoss which is the loss funcion for Sim-CLR training. When we train the model after sim-clr training, ContrastiveLoss class will calculate NTXentLoss and cross entropy loss with label smoothing, sum two value and return total loss value. At this point, we use NTXentLoss function from "https://github.com/sthalles/SimCLR/blob/master/loss/nt_xent.py"

## project1_arc
dd
