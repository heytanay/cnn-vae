# Deep Convolutional Variational Autoencoder
This Repository consists of a Deep Convolutional-Variational Autoencoder (CNN-VAE) made in Python using Tensorflow 1.x

Major Inspiration for implementation was taken from [Ha and Schmidhuber, "Recurrent World Models Facilitate Policy Evolution", 2018.
](https://worldmodels.github.io/#appendix).

Although currently a stand-alone implementation for general use, this model will be accompanied by a Mixed Density Recurrent Neural Network and Genetic Algorithms in my attempt to implement [this](https://arxiv.org/abs/1803.10122) paper presented at NeurIPS 2018.

## Installation
The main model has only 2 dependencies, Numpy and Tensorflow

For Numpy (On Linux/MacOS);
```
pip3 install numpy
```
OR
```
conda install numpy
```
For Numpy (On Windows);
```
pip install numpy
```
OR
```
conda install numpy
```

For Tensorflow (This model uses some deprecated function calls, so you need to have tensorflow version <= 1.5

On Linux/MacOS;
```
pip3 install tensorflow==1.12
```
OR
```
conda install tensorflow==1.12
```

On Windows;
```
pip install tensorflow==1.12
```
OR
```
conda install tensorflow==1.12
```
