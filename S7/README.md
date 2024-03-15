# Assignment 7

This assignment is all about getting the parameters down to 8k with an accuracy of 99.4% or more.

# Trial 1

We start with the first model where we only do batch normalisation and check the params and the accuracy. The model in trial 1 gets around 99.34 percent accuracy but it has 10K parameters

Only batch normalisation is used. Dropout is not used in this experiment.

Image augmentation is used for better training .

# Trial 2

The second model uses batch normalisation and drop out as well but with image augmentation.

The Test accuracy comes to 99.4 but the parameter size is 13k

The model takes an input image batch and passes it through a series of convolutional and pooling layers to extract features and reduce spatial dimensions.

Specifically, the input first goes through a convolutional block convblock1 to generate 16 feature maps. ReLU activation and batch normalization are applied to introduce non-linearity and stabilize training. Dropout is also applied to prevent overfitting.

The output then passes through a second convolutional block convblock2 to double the number of feature maps to 32. Similar activations, batch norm and dropout are also applied here.

Next is a transition block convblock3 that reduces the number of channels back down to 10, keeping the spatial dimensions unchanged.

After that, a 2x2 max pooling layer pool1 halves the height and width. This further reduces spatial redundancy to enable extracting more global features in subsequent layers.

The downsampled output goes through another series of 4 convolutional blocks with 16 channels each. ReLU, batch norm and dropout are again applied to improve training. Padding is added in the last block to maintain spatial size.

Finally, global average pooling gap aggregates the features across the spatial dimensions. This is fed into a final convolutional block convblock8 with 10 channels to generate output class scores. Log softmax activation provides a normalized probability distribution over the 10 classes that the model is trained to classify.

So in summary, this model takes input images, processes them through a series of convolutional, pooling and fully connected layers to generate class probability scores. The architecture is designed to minimize overfitting while extracting useful features for the classification task.

# Trial 3

In trial 3 we achieve 99.4% accuracy with a parameter size of less than 8k by:

reducing the channel size
drop out
regularisation
image augmentation
GAP

The overall architecture consists of:

Two convolutional blocks with BatchNorm and Dropout layers to extract spatial features from the input while regularizing the model. The convolutions reduce spatial dimensions while increasing channels.

A max pooling layer to downsample the feature maps.

A transition block to further reduce channels.

Another convolutional block with 3 conv layers to extract more complex features.

Global average pooling to convert the 2D feature maps into 1D by taking the spatial average.

A final conv block with 2 layers to transform the features into logits for the 10 classes.

The key components here are the convolutional blocks to hierarchically extract visual features, regularization methods like BatchNorm and Dropout to prevent overfitting, pooling to downsample spatially, and global average pooling to produce the final class logits.
