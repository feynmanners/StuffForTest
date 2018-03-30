import matplotlib.pyplot as plt
import numpy as np
import os
import collections
import tensorflow as tf

from tensor2tensor import models
from tensor2tensor import problems
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import registry
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry


@registry.register_model
class custom_model(t2t_model.T2TModel):

  def body(self, features):
    #print(2.1)
    if self._hparams.initializer == "orthogonal":
      raise ValueError("LSTM models fail with orthogonal initializer.")

    train = self._hparams.mode == tf.estimator.ModeKeys.TRAIN

    inputs = features["targets"]
    encoder_outputs=common_layers.flatten4d3d(inputs)
    #print(inputs)
    shifted_targets = common_layers.shift_right(inputs)
    final_encoder_state=None
    #size0,size1,size2,size3=tf.shape(shifted_targets)

    #I think embedding may be handled by problem

     # Flatten inputs.
    inputs = common_layers.flatten4d3d(shifted_targets)
    #rnn=RNN(hparams,train)
    # LSTM decoder
    #decoder_output, _ = lstm_attention_decoder(inputs, self._hparams, train, "decoder",final_encoder_state, encoder_outputs)
    #decoder_output = LSTM_custom(inputs, self._hparams, train, "decoder",final_encoder_state, encoder_outputs)[0]
    #decoder_output, _ = lstm(inputs, self._hparams, train, "decoder")
    decoder_output, _ = lstm_SA(inputs, self._hparams, train, "decoder")

    return tf.expand_dims(decoder_output , axis=2)

def lstm_SA(inputs, hparams, train, name, initial_state=None):
  """Run LSTM cell on inputs, assuming they are [batch x time x size]."""

  def dropout_lstm_cell():
    return tf.contrib.rnn.DropoutWrapper(
        tf.contrib.rnn.BasicLSTMCell(hparams.hidden_size),
        input_keep_prob=1.0 - hparams.dropout * tf.to_float(train))

  layers = [dropout_lstm_cell() for _ in range(hparams.num_hidden_layers)]
  with tf.variable_scope(name):
    inputs=self_attention(inputs,hparams, "attn")
    return tf.nn.dynamic_rnn(
        tf.contrib.rnn.MultiRNNCell(layers),
        inputs,
        initial_state=initial_state,
        dtype=tf.float32,
time_major=False)



def attention_fun(Q, K, scaled_=True):
  attention = tf.matmul(Q, K, transpose_b=True)  # [batch_size, sequence_length, sequence_length]

  if scaled_:
      d_k = tf.cast(tf.shape(K)[-1], dtype=tf.float32)
      attention = tf.divide(attention, tf.sqrt(d_k))  # [batch_size, sequence_length, sequence_length]

  attention = tf.nn.softmax(attention, dim=-1)  # [batch_size, time, time]
  return attention


def self_attention(inputs, hparams, name, initial_state=None):

  Q=tf.layers.dense(inputs,hparams.hidden_size,use_bias=False) # [batch_size, time, hidden_dim]
  K=tf.layers.dense(inputs,hparams.hidden_size,use_bias=False) # [batch_size, time, hidden_dim]
  V=tf.layers.dense(inputs,hparams.hidden_size,use_bias=False) # [batch_size, time, hidden_dim]

  attention=attention_fun(Q,K, scaled_=True)

  output=tf.matmul(attention,V)

  return output

