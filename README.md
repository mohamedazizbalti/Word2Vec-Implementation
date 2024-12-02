# Word2Vec Implementation with PyTorch

This repository contains an implementation of the Word2Vec model using PyTorch for generating word embeddings, The implementation uses the Skip-Gram model.

## Overview

The Word2Vec model is a shallow neural network used to learn distributed representations of words in a continuous vector space. This implementation uses the Skip-Gram architecture, where the goal is to predict context words given a target word.

This implementation includes:
- Training a Word2Vec model using custom text data.
- A custom Log Loss function for training the model.
- Support for training on either CPU or GPU (CUDA-enabled devices).
- Use of the Adam optimizer for efficient parameter updates.

## Features

- **Skip-Gram Model**: The Skip-Gram model is used to predict context words based on a given target word.
- **Log Loss**: A custom log loss function is used to optimize the model.
- **GPU Support**: The model can be trained on GPU if available, otherwise, it defaults to CPU.
- **Customizable Parameters**: You can customize the vocabulary size, embedding dimensions, and other training parameters.

## Requirements

To run this implementation, you will need the following libraries:

- Python 3.x
- PyTorch
- Numpy

You can install the required libraries.
