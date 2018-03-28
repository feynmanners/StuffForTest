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

# Enable TF Eager execution
from tensorflow.contrib.eager.python import tfe
tfe.enable_eager_execution()

# Other setup
Modes = tf.estimator.ModeKeys

# Setup some directories
data_dir = os.path.expanduser("~/t2t/data")
tmp_dir = os.path.expanduser("~/t2t/tmp")
train_dir = os.path.expanduser("~/t2t/train")
checkpoint_dir = os.path.expanduser("~/t2t/checkpoints")
tf.gfile.MakeDirs(data_dir)
tf.gfile.MakeDirs(tmp_dir)
tf.gfile.MakeDirs(train_dir)
tf.gfile.MakeDirs(checkpoint_dir)
gs_data_dir = "gs://tensor2tensor-data"
gs_ckpt_dir = "gs://tensor2tensor-checkpoints/"

# Fetch the MNIST problem
ptb_problem = problems.problem("languagemodel_ptb10k")



# Setup helper functions for encoding and decoding
def encode(input_str, output_str=None):
  """Input str to features dict, ready for inference"""
  inputs = encoders["targets"].encode(input_str) + [1]  # add EOS id
  batch_inputs = tf.reshape(inputs, [1, -1, 1])  # Make it 3D.
  return {"targets": batch_inputs}

def decode(integers):
  """List of ints to str"""
  integers = list(np.squeeze(integers))
  if 1 in integers:
    integers = integers[:integers.index(1)]
  return encoders["inputs"].decode(np.squeeze(integers))

# Copy the vocab file locally so we can encode inputs and decode model outputs
# All vocabs are stored on GCS
vocab_file = os.path.join(gs_data_dir, "vocab.lmptb.10000")
#!/home/rlmcavoy/.local/bin/gsutil cp {vocab_file} {data_dir}

# The generate_data method of a problem will download data and process it into
# a standard format ready for training and evaluation.
ptb_problem.generate_data(data_dir, tmp_dir)

# Get the encoders from the problem
encoders = ptb_problem.feature_encoders(data_dir)

# Now let's see the training MNIST data as Tensors.
ptb_example =tf e.Iterator(ptb_problem.dataset(Modes.TRAIN, data_dir)).next()
print(ptb_example)

# Create your own model

class MySimpleModel(t2t_model.T2TModel):

  def body(self, features):
    print(2.1)
    if self._hparams.initializer == "orthogonal":
      raise ValueError("LSTM models fail with orthogonal initializer.")
    
    train = self._hparams.mode == tf.estimator.ModeKeys.TRAIN
    
    inputs = features["targets"]
    
    shifted_targets = common_layers.shift_right(inputs)
    
    size0,size1,size2,size3=tf.shape(shifted_targets)
    
    #I think embedding may be handled by problem
    
     # Flatten inputs.
    inputs = common_layers.flatten4d3d(shifted_targets)
    # LSTM encoder.
    encoder_output, _ = lstm(
        tf.reverse(inputs, axis=[1]), self._hparams, train, "encoder")
    #xshape = tf.shape(encoder_output)
    
    #outputs = tf.reshape(encoder_output, [xshape[0], xshape[1] * xshape[2]])
    
    #outputs=tf.layers.dense(outputs, self._hparams.hidden_size, name="decoder")
    print(2.9)
    return tf.expand_dims(encoder_output , axis=2)

def lstm(inputs, hparams, train, name, initial_state=None):
  """Run LSTM cell on inputs, assuming they are [batch x time x size]."""

  def dropout_lstm_cell():
    return tf.contrib.rnn.DropoutWrapper(
        tf.contrib.rnn.BasicLSTMCell(hparams.hidden_size),
        input_keep_prob=1.0 - hparams.dropout * tf.to_float(train))

  layers = [dropout_lstm_cell() for _ in range(hparams.num_hidden_layers)]
  with tf.variable_scope(name):
    return tf.nn.dynamic_rnn(
        tf.contrib.rnn.MultiRNNCell(layers),
        inputs,
        initial_state=initial_state,
        dtype=tf.float32,
time_major=False)




hparams = trainer_lib.create_hparams("lstm_seq2seq", data_dir=data_dir, problem_name="languagemodel_ptb10k")
#hparams.hidden_size = 64
#hparams.num_hidden_layers = 1
#hparams.dropout = .2
#hparams.learning_rate = 20
model = MySimpleModel(hparams, Modes.TRAIN)

# Prepare for the training loop

# In Eager mode, opt.minimize must be passed a loss function wrapped with
# implicit_value_and_gradients
@tfe.implicit_value_and_gradients
def loss_fn(features):
  A, losses = model(features)
  return losses["training"]

# Setup the training data

#TODO: Have to use padding to allow batching 
BATCH_SIZE = 32
ptb_train_dataset = ptb_problem.dataset(Modes.TRAIN, data_dir)
print("output_shapes: ")
ptb_train_dataset.output_shapes['targets']= [200]

ptb_train_dataset = ptb_train_dataset.repeat(None).padded_batch(BATCH_SIZE,ptb_train_dataset.output_shapes)

optimizer = tf.train.AdamOptimizer()

# Train
NUM_STEPS = 1000

for count, example in enumerate(tfe.Iterator(ptb_train_dataset)):
  #targets is both the input and the output
  #print(tf.shape(example["targets"]))
  print(0)
  example["targets"] = tf.reshape(example["targets"], [BATCH_SIZE, -1, 1, 1])  # Make it 4D.
  #print(tf.shape(example["targets"]))
  print(1)
  loss, gv = loss_fn(example)
  print(2)
  optimizer.apply_gradients(gv)
  print(3)
  if count % 10 == 0:
    print("Step: %d, Loss: %.3f" % (count, loss.numpy()))
  if count >= NUM_STEPS:
    break
    
model.set_mode(Modes.EVAL)
ptb_eval_dataset = ptb_problem.dataset(Modes.EVAL, data_dir)

ptb_train_dataset = ptb_train_dataset.repeat(None).padded_batch(1,ptb_train_dataset.output_shapes)

# Create eval metric accumulators for accuracy (ACC) and accuracy in
# top 5 (ACC_TOP5)
metrics_accum, metrics_result = metrics.create_eager_metrics(
    [metrics.Metrics.NEG_LOG_PERPLEXITY, metrics.Metrics.APPROX_BLEU])

for count, example in enumerate(tfe.Iterator(ptb_eval_dataset)):
  if count >= 200:
    break

  # Make the inputs and targets 4D
  example["targets"] = tf.reshape(example["targets"], [1, -1, 1, 1])
  #example["targets"] = tf.reshape(example["targets"], [1, 1, 1, 1])

  # Call the model
  predictions,_ = model(example)
  #predictions = decode(output)

  # Compute and accumulate metrics
  metrics_accum(predictions, example["targets"])

# Print out the averaged metric values on the eval data
for name, val in metrics_result().items():
  print("%s: %.2f" % (name, val))