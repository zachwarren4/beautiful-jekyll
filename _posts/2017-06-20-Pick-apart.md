---
layout: post
title: "A novice picking apart a large conv net"
date: 2017-06-20
---

As with most aspects of programming, I've quickly been able to Google my way through writing my first couple neural nets of various size and difficulty. However, unlike other topics I've encountered, I likely will never take a course that teaches me exactly what is going on with each function call and parameter setting. Today, I decided I'm going to dissect the large convolutional neural net written in Python with Keras from [this](http://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/) tutorial I mentioned yesterday. I'm going to layout the code here and try to explain what every line is doing, Googling what I don't know. I'm going to start at the very beginning, with the import statements.

## Imports

```python
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
```
The first two lines, ```import numpy``` and ```from keras.datasets import mnist``` are rather straight forward. Numpy is the 'number' package that, at its simplest, arranges numbers in contiguous memory blocks, more of a traditional array type vs python's default lists. In order to do this, all numpy array values must be of the same type (so they are the same size) and we lose the flexibilty of python's lists. The benefits, however, is a gigantic speedup, without which would render python pretty much useless for large mathematical computations. The second line is simply importing the MNIST dataset.

Next, we import the *Sequential* model from Keras. I'll get to the details of the model when it comes time to create it but for now, the Sequential model is just an object made by Keras that allows layers to be added...sequentially.

Now for the layers:

* Dense: Keras documentation says it is 'just your regular densely-connected NN layer', which doesn't help me much. What does make sense is this equation![act-eq](/img/activation-eq.png). This equation simply defines an output of a neuron based on the inputs *x*, a weight vector associated with them *w*, a bias associated with the neuron *b*, and the *sigmoid function* described in [this](http://neuralnetworksanddeeplearning.com/chap1.html) page. 
