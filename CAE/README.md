# Class-Project-IFT-6266
Class project for IFT6266 : Conditional Image Generation


In this repository I will store the codes necessary for the class project.

The 'lib' directory contains Layers (convolutional, LSTM, ..) and Dataset (loading images,..) utilities shared by all codes.

The 'code' directory contains the different algorithms I'll be using.

### Layers :
Here there are many classes needed to build the networks. Each class corresponds to a given layer and they are coded such that we can stack them easily. Everything is commented and described in the class docs.
The Tool class is a utility class containing many different functions from either lasagne or theano. It is somewhat repetitive but I did this for sake of completude.

### Img : 
Here there is a single class whose methods are used to load mini-batches of data, plot results and save images.

### Codes : 
The description of the codes and their results is in the blog : https://ift6266mcomin.wordpress.com/
