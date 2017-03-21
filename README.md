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
