Going deeper with convolutions

Szegedy, Christian, et al. "Going deeper with convolutions." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.

https://arxiv.org/pdf/1409.4842v1.pdf

*Keypoints*

1. Use 1x1 convolutinal layers

Reductin number of channel  --> Reduced computation cost

ex) Assume that without bias

Input feature (28x28x128) -> 1x1 conv (32 filters)-> 3x3 conv (64 filters) : 28x28x32x1x1x128 + 28x28x64x3x3x32 ~= 1.7M
Input feature (28x28x128) -> 3x3 conv (64 filters) : 28x28x64x3x3x128 ~= 5.7M

2. Inception Module

Before the description of inception, we should look at the typical problems of CNN in the image recognition problem.
In order to have a serviceable level of recognition rate using deep learning, it is necessary to learn a large amount of data. In the architecture of deep learning (CNN Architecture), the view is that the recognition rate is good when the layers are wide (there are many neurons) and deep (there are many layers). However, deep learning has always been difficult to break away from problems such as learning time and computation speed along with problems such as overfitting and vanishing gradient problems in this situation.

Papers suggesting solutions such as residual learning (skip-connection) or LSTM (Highway network) to solve this problem have always given us great benefits in improving the performance of the network.

One of these methods, the inception module (+ Auxiliary classifier), is introduced.

![InceptionModule](https://user-images.githubusercontent.com/96281316/148023896-209488a1-d0bf-4ece-bcb0-144bc4a63be8.png)

Looking from the naive version, various convolutions are applied, and the result value that has been calculated in the conv layer and the result value using maxpooling are all synthesized and concatenated as an output.

Although there are dimension reductions, 1x1 conv layer is applied to the existing naive version of the layer, keeping the height and width constant, and reducing the number of channels has the advantage of reducing the amount of computation.

The point to note is that 1x1, 3x3, and 5x5 convolution operations are performed respectively, and the maximum spare connection is maintained in the process of extracting the feature map, and an effort was made to make it as dense as possible in matrix operation. However, in order to reduce the computation cost that can be too large, a 1x1 conv layer is applied before or after each layer.

[Concatenation --> Apply axis=3 (apply channel)]

3. Global Average Pooling

For use as an existing classifier layer, a model that predicts the probability of the result value is used by putting a fully connected layer at the end of the convolutional layer operation. However, the method used in this paper pointed out the problems of these methods and tried to effectively solve this problem by using GAP (Global Average Pooling).

First, looking at the problems that occur in the use of the previous FC layer, the input value was made into a very long vector using the flatten layer, and the vector was received from the fc layer and mapped to each class to classify it. In this process, spatial information was also lost, and a lot of weights were required, so the amount of computation increased dramatically.

The process of GAP is simple. In the final processing of the convolutional layer, the number of feature maps is generated as many as the number of classes to be classified. This means that the number of filters of the last conv layer should be the same as that of classes.

In this way, the number of parameters can be greatly reduced, and at the same time, the effect of preventing overfitting can be seen. As a slight drawback, since all values of the image are used as an average, information loss may follow, but it is used because of its good performance nonetheless.

![Global Average Pooling](https://user-images.githubusercontent.com/96281316/148026655-3938c837-8bde-4944-ae0c-aaa7ed155dad.png)

figure by [https://www.researchgate.net/figure/Difference-between-fully-connected-layer-and-global-average-pooling-layer_fig2_339096868]

4. Auxiliary classifier

Auxiliary classifier is a concept first introduced in googlenet (inception-net v1), and vanishing gradients occur in the sense that it is difficult to reach the initial stage during the backpropagation process as the gradient gets deeper. Due to this problem, the softmax classifier is placed in the middle during training to enable backpropagation in the middle. This is a concept that allows the gradient to be transmitted well throughout the model.

A point to be aware of when using the auxiliary classifier is to multiply the auxiliary classifier by 0.3 to avoid having a large influence on the weight value when using backpropagation to reduce the effect.
Another is a technique used to solve the vanishing gradient problem, a problem that occurs during training, so all Auxiliary classifiers in the middle process should be removed and only the last classifier should be used for testing.

*Structure*

![GoogleNet](https://user-images.githubusercontent.com/96281316/148027586-0b38c140-1518-4949-b867-a1bf18451b6b.png)
