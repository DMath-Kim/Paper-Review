Very Deep Convolutional Networks for Large-Scale Image Recognition [2015]
- Karen Simonyan, Andrew Zisserman

*Key Points* [+ Because only 3x3 filter was used, the depth could be increased.]

1. Compared to the previous AlexNet, we doubled the depth of the model by using a 3x3 filter. (All conv layer using 3x3 filter size)

In contrast to the 11x11 filter or the 7x7 filter, the 3x3 filter can produce the effect of a 7x7 receptive field when stride is 1 and 3 conv layers are repeated.

Difference between 3x3 filter and 7x7 filter

- Increase the nonlinearity of the decision function

Assuming that the activation function is applied once for the conv layer, if 3x3 is applied three times, the activation function, which is a non-linear function, is applied three times. Therefore, as the layer increases, the nonlinearity increases, which can lead to an increase in the feature identification of the model.

- Reduction Learning parameters

7x7 1 filter -> 49 params, 3x3 3 filters -> 27 params

*New Approach*

- Weight initialization 

1) A relatively shallow 11-Layer network is first trained. At this time, the weights are initialized to random values to follow a normal distribution.

2) When the learning is completed to some extent, the weights of the four layers of the input layer and the last three fully-connected layers are used as the initial values of the network to be trained.


*Structure Table*

![VGGNet](https://user-images.githubusercontent.com/96281316/147906460-448ef55c-4628-42a9-a295-b91beda1a81f.png)
