# How_to_make_a_language_translator
This is the code for "How to Make a Language Translator - Intro to Deep Learning #11' by Siraj Raval on YouTube

# How to get this working:

Change the order of the arguments in `sampled_softmax_loss` in `tensorflow/python/ops/nn_impl.py` on line 1123.

```
def sampled_softmax_loss(weights,
                         biases,
                         labels,
                         inputs,
                         ...
```

to

```
def sampled_softmax_loss(weights,
                         biases,
                         inputs,
                         labels,
                         ...
```

Also in TensorFlow, this time in the `sequence_loss_by_example` function in `tensorflow/contrib/legacy_seq2seq/python/ops/seq2seq.py`, change line 1067 from

```
crossent = softmax_loss_function(target, logit)
```

to

```
crossent = softmax_loss_function(logit, target)
```


and change line 103 of `seq2seq_model.py` from

```
def sampled_loss(labels, inputs):
```

to

```
def sampled_loss(inputs, labels):
```

Also, change

```
tf.nn.rnn_cell
```

to

```
tf.contrib.rnn.core_rnn_cell
```

and change

```
tf.nn.seq2seq
```

to

```
tf.contrib.legacy_seq2seq
```

and change

```
from tensorflow.models.rnn.translate import data_utils
```

to

```
import data_utils
```

and change

```
self.saver = tf.train.Saver(tf.global_variables())
```

to

```
self.saver = tf.train.Saver(tf.all_variables())
```

Finally, in `translate.py`, change

```
s_train, t_train, s_dev, t_dev, _, _, _, _ = data_utils.prepare_data(FLAGS.data_dir, FLAGS.s_vocab_size, FLAGS.t_vocab_size, source, target)
```

to

```
s_train, t_train, s_dev, t_dev, _, _ = data_utils.prepare_data(FLAGS.data_dir, FLAGS.s_vocab_size, FLAGS.t_vocab_size, source, target)
```

and on line 106, change

```
session.run(tf.initialize_all_variables())
```

to

```
session.run(tf.global_variables_initializer())
```

FINALLY, run `python3 translate.py en fr` to translate from English to French.

With all these changes, this worked with the latest versions of Python 3 and TensorFlow.

## Overview

This is the code for [this](https://youtu.be/nRBnh4qbPHI) video on Youtube by Siraj Raval as part of the Deep Learning Nanodegree course with Udacity. This code implemnents [Neural Machine Translation](https://github.com/neubig/nmt-tips) which is what Google Translate uses to translate Languages.

## Dependencies

* tensorflow
* nltk
* six

Install missing dependencies with [pip](https://pip.pypa.io/en/stable/)

To train model on data and test it to compute the [BLEU score](https://en.wikipedia.org/wiki/BLEU) run this:

``python translate.py source_language target_language`` (i.e. python translate.py fr en for fr->en translation)

## Credits

The credits for this code (modulo the changes discussed above) go to [Siraj Raval](https://github.com/llSourcell/How_to_make_a_language_translator) for pointing me towards this example and [the TensorFlow team](https://www.tensorflow.org/tutorials/seq2seq) at Google, for writing both the code and the documentation.
