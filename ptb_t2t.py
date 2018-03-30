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
ptb_example =tfe.Iterator(ptb_problem.dataset(Modes.TRAIN, data_dir)).next()
print(ptb_example)

# Create your own model

class MySimpleModel(t2t_model.T2TModel):

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
    size0,size1,size2,size3=tf.shape(shifted_targets)
    
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
#    return tf.nn.static_rnn(
#        tf.contrib.rnn.MultiRNNCell(layers),
#        [inputs],
#        initial_state=initial_state,
#        dtype=tf.float32)
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


def LSTM_SA_custom(inputs, hparams, train, name, initial_state,
                           encoder_outputs):
  """A static RNN.
  Similar to tf.nn.static_rnn
  """
  def dropout_lstm_cell():
    return tf.contrib.rnn.DropoutWrapper(
        tf.nn.rnn_cell.BasicLSTMCell(hparams.hidden_size),
        input_keep_prob=1.0 - hparams.dropout * tf.to_float(train))


  layers=[dropout_lstm_cell() for _ in range(hparams.num_hidden_layers)]
  

  batch_size = int(inputs.shape[0])
  inputs=self_attention(inputs,hparams, "attn")
  for c in layers:
    state = c.zero_state(batch_size, tf.float32)
    outputs = []
    
    input_seq = tf.unstack(inputs, num=int(inputs.shape[1]), axis=1)
    for inp in input_seq:
      output, state = c(inp, state)
      outputs.append(output)

    input_seq = tf.stack(outputs, axis=1)
  
    # Returning a list instead of a single tensor so that the line:
    # y = self.rnn(y, ...)[0]
    # in PTBModel.call works for both this RNN and CudnnLSTM (which returns a
    # tuple (output, output_states).
  return [input_seq]

  def _add_cells(self, cells):
    # "Magic" required for keras.Model classes to track all the variables in
    # a list of Layer objects.
    # TODO(ashankar): Figure out API so user code doesn't have to do this.
    for i, c in enumerate(cells):
      setattr(self, "cell-%d" % i, c)
    return cells 
   
  
hparams = trainer_lib.create_hparams("lstm_attention", data_dir=data_dir, problem_name="languagemodel_ptb10k")
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


  example["targets"] = tf.reshape(example["targets"], [BATCH_SIZE, -1, 1, 1])  # Make it 4D.
  #print(tf.shape(example["targets"]))
  #print(1)
  loss, gv = loss_fn(example)
  #print(2)
  optimizer.apply_gradients(gv)
  #print(3)
  if count % 10 == 0:
    print("Step: %d, Loss: %.3f" % (count, loss.numpy()))
  if count >= NUM_STEPS:
    break
    
model.set_mode(Modes.EVAL)
ptb_eval_dataset = ptb_problem.dataset(Modes.EVAL, data_dir)

#b_train_dataset = ptb_train_dataset.repeat(None).padded_batch(1,ptb_train_dataset.output_shapes)

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
