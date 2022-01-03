Deep Residual Learning for Image Recognition [2015]
- Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun

*Key Points % New Approach*

Residual block (skip connection)

![Residual learning](https://user-images.githubusercontent.com/96281316/147909773-f3d2b45b-f92e-437d-a281-fbb1dfee766d.png)

- Degradation Problem [Using Residual Block]

This is a method designed to solve the problem of vanishing gradients that occur when implementing very deep neural networks.

- Residual Learning

Instead of hoping each few stacked layers directly fit a desired underlying mapping, we explicitly let these layers fit a residual mapping.

y = F(X, {W_i}) + W_s*X

W_s -> The identity mapping is multiplied by a linear projection W to expand the channels of shortcut to match the residual.

*How to solve the vanishing gradient problem*

Most of the vanishing / exploding gradient problems can be solved through normalized initialization and intermediate normalization layers, and networks in several dozen layers can be converged through backpropagation.

However, in order to solve the gradient vanishing problem that occurs in deep networks, methods of residual learning and concatenation densely connected architectures are introduced.

skip connection (shortcut connection) method only increases the amount of addition operation and does not significantly affect the number of learning parameters. Deep models with a larger number of layers are also well trained and have the advantage of being able to learn more easily and quickly. Also, if you think about it a little, since the identity mapping of the previous layer is coming in as a signal, the vanishing gradient does not occur even if the layer gets deeper.

*Structure*

![ResNet](https://user-images.githubusercontent.com/96281316/147911253-b8024e6d-0faf-4c09-8cf3-3d68a4f80ae7.png)

- Next Residual Block (depth * 2)
