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
The first two lines, `import numpy` and `from keras.datasets import mnist` are rather straight forward. Numpy is the 'number' package that, at its simplest, arranges numbers in contiguous memory blocks, more of a traditional array type vs python's default lists. In order to do this, all numpy array values must be of the same type (so they are the same size) and we lose the flexibilty of python's lists. The benefits, however, is a gigantic speedup, without which would render python pretty much useless for large mathematical computations. The second line is simply importing the MNIST dataset.

Next, we import the *Sequential* model from Keras. I'll get to the details of the model when it comes time to create it but for now, the Sequential model is just an object made by Keras that allows layers to be added...sequentially.

Now for the layers:
* Dense: Keras documentation says it is 'just your regular densely-connected NN layer', which doesn't help me much. What does make sense is this equation![act-eq](/img/activation-eq.png). This equation simply defines an output of a neuron based on the inputs *x*, a weight vector associated with them *w*, a bias associated with the neuron *b*, and the *sigmoid function* described in [this](http://neuralnetworksanddeeplearning.com/chap1.html) page. So the dense layer really is just the 'regular' layer of a NN.

* Dropout: The dropout layer, according to Keras docs, seems pretty genius. Based on [this](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf) paper, the dropout layer does exactly what you'd expect it to, it drops out inputs based on some *rate* each 'update during training time,' which I'm going to assume means something in the backend (which I will get to at a later date). Dropping out some inputs prevents *overfitting*, a situation caused by, based on my best understanding, the model getting too 'familiar' with the training data. The dropout 'layer' has no input or output, it just applies the Dropout scheme to the input.

* Flatten: Changes the output shape of the model to a 'flat', one dimensional array.

* Conv2D: Here, I really am stepping into unknown territory. The 2D is fairly self-explanatory, we're using image data and naturally use the 2D convolution layer to deal with that. Convolutional neural networks are murky waters for me. Something that didn't help at all, but that I found cool was the Wikipedia [page](https://en.wikipedia.org/wiki/Convolutional_neural_network) on ConvNets that says they are 'inspired by the animal visual cortex.' Fun fact at least. After further investigation, I'm going to leave this alone until I finish reading the [textbook](http://neuralnetworksanddeeplearning.com/) I keep mentioning as the last chapter is a thorough introduction to ConvNets.

* MaxPooling2D: Similarly, I'm going to leave this alone as it is attached to the convolutional layer.

The `np_utils` mentioned in the second to last line is actually just a file with two functions, `to_categorical(y, num_classes)` and `normalize(x, axis, order)`. I'll look into the first function when it's used later.

Finally, the `from keras import backend as K` imports the backend module based on the specified backend. Keras allows either *tensorflow*, *Theano*, or *CNTK*, which is Microsoft's open-source toolkit that I did not know existed until today. 

That covers all of the imports, which really covers a lot of the actual code as well as far as basic understanding goes. But I started this mission, so I'm going to finish it.

## Setup

There are three lines of code that do some setup that I felt worth investigating. Here they are, in all of their glory:

```python
K.set_image_dim_ordering('th')
seed = 7
numpy.random.seed(seed)
```

Not much but fairly important. The first line sets the shape of the convolutional kernels, which I will have to get back to but both for myself and anyone else, there's a good explanation [here](https://stackoverflow.com/questions/39547279/loading-weights-in-th-format-when-keras-is-set-to-tf-format). The next two lines are there to make the results reproducible. By setting the seed for the random number generator, we can do the same thing again without varying results.

## Loading and Shaping the Data

Now we load the MNIST dataset and reshape it appropirately.

```python
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

X_train = X_train / 255
X_test = X_test / 255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
```

The first line loads the data from the Keras module 'mnist' into two tuples, one that has the training data, one with the testing data. The tuples are (training, target), the data the model trains on and then the 'answers.'

In the more basic model, we reshaped the data to be one 784 length vector for each image. Since we changed the input layer to Conv2D, it's expecting a *pixel x width x height* dimension matrix, since our images are *1 x 28 x 28*, we just need to reshape X_train and X_test to be multidimensional arrays of length 60,000 and 10,000 respectively with the above dimensions for the images inside. If these were RGB images, there would be 3 values per pixel so the dimensions would be *3 x 28 x 28*. We change the type to float32, as that's both the default for Keras and the next two lines are going to normalize the data which will leave us with float values *0 - 1*.

The `to_categorical()` lines transforms the 'answers' into something useful. The [guide](http://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/) I used explains it as one hot encoding, which took me back to taking Digital Logic Sophomore year. Essentially, it turns our target data into a binary matrix with the *1* in the appropriate place for that integer. For example, if the target of something in *X_train* was supposed to be a 5, the corresponding *y_train* data would look like *[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]*. 

`num_classes` is just the number of possible outputs, in our case, 10.

## Creating the model

