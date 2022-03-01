# PyTorch: Fully Connected Autoencoder

In this repo we implement a fully connected autoencoder using PyTorch.

## Overview

Given a sensible data set, autoencoder are able learn a lower-dimensional representation of the data which we call the "code" (representation). Thus, they can be used to discover and identify intrinsic and significant features present in the data set.

The following implementation an autoencoder uses fully connected layers. Among others, network shape (depth and widths), activation function and loss function are adjustable parameters. Furthermore, CUDA support is provided by default.

The class FullyConnectedAutoencoder defines the following methods:

* __init__ (constructor)
* encode (maps from input space to code space)
* decode (maps from code space to output space)
* forward (concatenation of encode and decode)
* train_model (method to train the network)
* save_model (saves model as *.pt)
* load_model (load model from *.pt)

## Dependencies

 1. numpy
 2. torch
    *  torch.optim
    *  torch.nn
