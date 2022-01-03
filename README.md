Gradient-Based Learning Applied to Document Recognition (LeNet-5) [1998]
- Yann LeCun Leon Bottou Yoshua Bengio and Patrick Haner

Key Points

1. Local Receptive Fields

In the case of the Fully Connected Layer, which was used before the introduction of the Convolutional Neural Network, it is not useful to receive all input information of the previous layer. It is more important to understand the characteristics in a specific space than to receive and use all the information of the previous layer.

2. Sub-Sampling

Sub-sampling is designed to reduce the dependence on precise positioning in the feature map generated by the conv layer in CNN.

Reliance on specific feature positioning has a disadvantage in network construction for input data that has undergone affine transformation. Therefore, the positioning in the feature map is not the exact position of the input, but the position relative to other features in the feature map.

Reducing this dependence on exact location leads to reducing spatial resolution using a conv layer.

3. Weight Sharing

Since the spatial information of the previous feature map is shared, the weights are related to each other.

New Approach
1) Reduced parameters using convolutional layer
2) Preserved the topology of input data
