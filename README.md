# cifar10-vgg16

## Description
CNN to classify the cifar-10 database by using a vgg16 trained on Imagenet as base.
The approach is to transfer learn using the first three blocks (top layers) of vgg16 network and adding FC layers on top of them and train it on CIFAR-10. 

## Training
Trained using two approaches for 250 epochs:
1. Keeping the base model's layer fixed, and
2. By training end-to-end

## Model Summary
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv_bn_relu (ConvBNRelu)    (None, 32, 32, 64)        2048      
    _________________________________________________________________
    conv_bn_relu_1 (ConvBNRelu)  (None, 32, 32, 64)        37184     
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 16, 16, 64)        0         
    _________________________________________________________________
    conv_bn_relu_2 (ConvBNRelu)  (None, 16, 16, 128)       74368     
    _________________________________________________________________
    conv_bn_relu_3 (ConvBNRelu)  (None, 16, 16, 128)       148096    
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 8, 8, 128)         0         
    _________________________________________________________________
    conv_bn_relu_4 (ConvBNRelu)  (None, 8, 8, 256)         296192    
    _________________________________________________________________
    conv_bn_relu_5 (ConvBNRelu)  (None, 8, 8, 256)         591104    
    _________________________________________________________________
    conv_bn_relu_6 (ConvBNRelu)  (None, 8, 8, 256)         591104    
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 4, 4, 256)         0         
    _________________________________________________________________
    conv_bn_relu_7 (ConvBNRelu)  (None, 4, 4, 512)         1182208   
    _________________________________________________________________
    conv_bn_relu_8 (ConvBNRelu)  (None, 4, 4, 512)         2361856   
    _________________________________________________________________
    conv_bn_relu_9 (ConvBNRelu)  (None, 4, 4, 512)         2361856   
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 2, 2, 512)         0         
    _________________________________________________________________
    conv_bn_relu_10 (ConvBNRelu) (None, 2, 2, 512)         2361856   
    _________________________________________________________________
    conv_bn_relu_11 (ConvBNRelu) (None, 2, 2, 512)         2361856   
    _________________________________________________________________
    conv_bn_relu_12 (ConvBNRelu) (None, 2, 2, 512)         2361856   
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, 1, 1, 512)         0         
    _________________________________________________________________
    flatten (Flatten)            (None, 512)               0         
    _________________________________________________________________
    dropout_13 (Dropout)         (None, 1, 1, 512)         0         
    _________________________________________________________________
    dense (Dense)                (None, 512)               262656    
    _________________________________________________________________
    batch_normalization_13 (Batc (None, 512)               2048      
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                5130      
    _________________________________________________________________
    activation (Activation)      (None, 10)                0         
    =================================================================
    Total params: 15,001,418
    Trainable params: 14,991,946
    Non-trainable params: 9,472
    _________________________________________________________________

## Hyper parameter
    training_epochs = 250
    batch_size = 128
    learning_rate = 0.1
    momentum = 0.9
    lr_decay = 1e-6
    lr_drop = 20



#### Files
Source Files:

- vgg16.py
    - load_images() : load cifar-10 images (train, test)
    - normalization() : normalization cifar-10 images
    - ConvBNRelu : create conv layer with relu,  batchnorm
    - VGG16Model  : create deep learning model based vgg16
    - train() : train VGG16Model with cifar-10 images
    - main() : main function that Initial images and model then, call train function
    
    
- cifar10vgg_custom.h5 : trained model's weights


|Model|Validation Accuracy
|:------:|:---:|
|[VGG-16](https://github.com/SeHwanJoo/cifar10-vgg16)|93.15%|
|[ResNet-20](https://github.com/SeHwanJoo/cifar10-ResNet-tensorflow)|91.52%|
|[ResNet-32](https://github.com/SeHwanJoo/cifar10-ResNet-tensorflow)|92.53%|
|[ResNet-44](https://github.com/SeHwanJoo/cifar10-ResNet-tensorflow)|93.16%|
|[ResNet-56](https://github.com/SeHwanJoo/cifar10-ResNet-tensorflow)|93.21%|
|[ResNet-110](https://github.com/SeHwanJoo/cifar10-ResNet-tensorflow)|93.90%|

