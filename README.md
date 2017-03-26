# deep_learning_nd_projects
Projects from the Udacity Deep Learning Nanodegree

## deep_jazz

Original repo: https://github.com/jisungk/deepjazz

Note: to get it to work, had to have a particular version of Theano installed.

Run `pip3 uninstall theano`, and then run

`pip3 install git+ssh://git@github.com/Theano/Theano.git@rel-0.9.0rc3`

Finally, change Keras so it uses a Theano backend. You can do this by changing
the `keras.json` file in the `~/.keras` directory, so that the key `backend`
has the value `theano`.

Finally, following the instructions in the repo above, run

`python generator.py [# of epochs]`.

I don't want to spoil what happens when you do this.

## TensorBoard

Original YouTube Video: https://www.youtube.com/watch?v=eBbEDRsCmv4

Instructions: navigate to `tensorboard`. Run `python tensorboard.py`. Then run
`tensorboard --logdir ./tmp/mnist_demo/<number of run>` and go to http://127.0.0.1:6006/ to
see the TensorBoard dashboard.

Running `python tensorboard_names.py` gives a cleaner board because the key
variables have been named.

Running `python tensorboard_param_search.py` creates the data to look at a board
that shows different results for different parameter settings.

### Visualizer

https://github.com/llSourcell/Tensorboard_demo

There's a great, clear example in the `tensorboard_visualizer` folder. Navigate
to the folder, and simply run `python complex.py`. Then, type `tensorboard --logdir=./tmp/mnist_logs` in the Terminal, and navigate to
http://127.0.0.1:6006/ to see the Tensorboard.

## 01_first_neural_network

https://github.com/udacity/deep-learning/tree/master/first-neural-network

Use a neural net, written from scratch using numpy, to predict bike sharing usage over time. This highlights the flexibility of neural nets - they are used here to build a very good model predicting bike sharing over time, capturing the complex, nonlinear relationships between the features and the output nicely.

The `DLND Your first neural network` notebook contains the code for building and the results from running the neural net. 
