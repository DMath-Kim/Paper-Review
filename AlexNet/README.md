ImageNet Classification with Deep Convolutional Neural Networks [2012]
- Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton

https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf

*Key Points*

1. The FC layer was tuned to distinguish a very large number of classes.
2. The depth of the layer increased.

*New Approach*

1. ReLU Nonlinearity

The ReLu activation function was used in the tanh function. With these changes, the CIFAR10 dataset reached up to 25% error 6 times faster.

2. Multiple GPUs

Multiple GPUs were used because it was difficult to process data sets with a wide variety of classes due to the low performance of the past GPUs.

3. Overlapping Pooling

Stride size is used smaller than kernel size.

Reduction of error and reduction of overfitting are the main advantages.

In the case of the previous LeNet-5, Average Pooling was used, but in this paper, MaxPooling was adopted.

4. Overfitting Problem

-Data Augmentation

-Dropout

*Structure*

![AlexNet](https://user-images.githubusercontent.com/96281316/147901124-31ea7ca9-1bb1-41c2-a2d2-4c09ef393ad0.png)
