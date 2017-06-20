---
layout: github-markdown
title: "Basic Neural Nets and Analysis"
date: 2017-06-19
---

Apologies for the delay since last post, had a couple weeks of vacation before getting starting for the summer. Last week, I spent some time getting some basic neural nets based on the handwriting (MNIST) dataset up and running. I followed [this](http://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/) guide to get started. 

## Setup

Most of the hours spent last week were getting my system setup to best use Tensorflow, Keras, CUDA, and CUDNN. I'm running Arch Linux 64 bit, which is a flavor of Linux with massive community support but a small user base so most big developers do not take it into consideration in their installation guides. Nonetheless, I hit the web and after many hours of breaking and unbreaking, copying files around, setting up environments and deleting them (over and over), I managed to get Tensorflow working with CUDNN on my GPU. The pay off was well worth it.

Prior to using my GPU, one epoch of training was taking ~4s on the comparatively simple and tiny 60,000 image handwriting dataset. With the GPU, that number dropped to near zero. After I entered into convolutional NN territory, the GPU was taking ~4s per epoch while the CPU was at around ~120s. Hours spent in setup paid dividends almost immediately.

## Preliminary Results

After getting everything up and running, I wanted to do a little bit of experimentation with the setup of the NN. Mainly, I was concerned with accuracy vs training time, which roughly translates into number of epochs for a small training set. I decided just to view that simple comparison over 10, 50, and 100 epochs of training. Here are the results:

![ten-epoch](/images/ten-epoch-acc.jpg)

![fifty-epoch](/images/fifty-epoch-acc.jpg)

![100-epoch](/images/100-epoch-acc.jpg)

A couple of interesting things I noted were that:

* Training was near perfect at around the tenth epoch, after that, the test set never really had much of an improvement.
* There were a couple of random dips, which I can only assume comes from the net adjusting a weight/bias on a more important or couple of more important neurons that had unintended consequences.
* On a couple of examples I didn't save, my models actually overfit and got less accurate as time went on. I've learned since that this is due to 'memorization' by the model, basically, the model is very good at recognizing the *exact* digits in the training set but then has trouble with new ones. 

A question I was asked by my professor was 'why is the model so accurate before the first epoch even starts?' After look around a little bit, it turns out that the initialization of the model just did a decent job. 

Here is the code:
```python
model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
model.add(Dense(num_classes, kernel_initializer='normal',activation = 'softmax'))
```

Each line is adding one layer to the net, the first adding a layer consisting of 768 neurons, one for each pixel initialized with a normal distribution of weights and biases. The second adding an output later consisting of ten neurons, one for each 'class' of output with the same type of initialization. The activations are the functions for activation on each layer. I'll be reading up on that further. 

## Conclusion and next steps

From here, I'll be working on getting some more models based on the MNIST dataset before I start working on the galaxy classification net, which will be much more resource intensive. 
