# CS_492_CV
CS_492 : semi-supervised learning for naver fashion dataset

In this repository, There are two folders: project1, project1_arc

## project1
> Project1 was created for two purposes. The first is to train the model for 300 epochs using only sim-clr Loss function. This pre-trained model was used in subsequent experiments. The second is The second is to train the previously pre-trained model with sim-clr loss and standard cross entropy loss. When calculating cross entropy loss, we applied label smoothing.

### ImageDataLoader.py
> When training time, we need to apply two differnet transforms on same images for labeled and unlabeled data and get its label of labeled data. When valid and test time, we need single image and its label. To do this, __getitem__ function will return images and label depend on data type. 

### loss.py
> ContrastiveLoss class calculates combined loss value during training time. When we train the model with only sim-clr loss, ContrastiveLoss class will calculate only NTXentLoss which is the loss funcion for Sim-CLR training. When we train the model after sim-clr training, ContrastiveLoss class will calculate NTXentLoss and cross entropy loss with label smoothing, sum two value and return total loss value. At this point, we use NTXentLoss function from "https://github.com/sthalles/SimCLR/blob/master/loss/nt_xent.py"

### models.py
> contrastive_model module is consist of feature extractor and linear layer for feature embedding and linear layer for classifition. We use efficientnet-b0 as feature extractor.

### main.py
> When training with only sim-clr loss, we removed comment sign('#') in line 155 to 157 and add comment sign in line 160 to 164. When training with combined loss, we executed current code. After training with only sim-clr loss, enter its checkpoint name and its session on ckpt and session parameter.


## project1_arc
> Project1_arc was created for introducing arcface loss and entropy minimization loss function.

### ImageDataLoader.py
> Same as the implementation in project1.

### loss.py
> ContrastiveArcLoss class was added. When we train the model after sim-clr training, ContrastiveArcLoss class will calculate NTXentLoss and arcface loss (applying cross entropy loss to pre-processed pred value) and entropy minimization loss.

### model.py
> ArcMargin module was added and weights_init_classifier was changed since last classification layer has no bias. To calculate arcface loss at training time, we need to do additional process to model prediction which is the output of the classifier layer. However, at test time, we need to get original model prediction value to calculate accuracy. Therefore, we added ArcMargin module to update parameter of classifier layer. After updating ArcMargin, copy the value of params into classifier of original model.

### main.py
> We conducted several experiments while changing the value of some parameters.


## The path of pre-trained model in NSML
1. pre-trained model with only sim-clr loss for 300 epochs
  - kaist006/fashion_dataset/51/c_model_e299
  
2. 
