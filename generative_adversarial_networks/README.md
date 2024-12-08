# Implementation of Generative Adversarial Networks

Generator and discriminator were not training at first, thus the issue of not having any updates in error. 
Error was stuck at 0.6931, which is equivalent to -ln(0.5), equivalent to random guessing of whether the input to the discriminator was from the dataset or generated by the generator.

Using the LeakyReLU activation function instead of the ReLU activation function, allowing gradients to propagate even when x < 0, compared to ReLU which makes the gradient go to zero when there are many negative inputs (x).