"""
References:
- YouTube video: How to Generate Music with Tensorflow (LIVE) by Siraj Raval
  https://www.youtube.com/watch?v=pg9apmwf7og

- GitHub repository: llSourcell/How_to_generate_music_in_tensorflow_LIVE (@llSourcell)
  https://github.com/llSourcell/How_to_generate_music_in_tensorflow_LIVE

- GitHub repository: bhaktipriya/Blues (bhaktipriya)
  https://github.com/bhaktipriya/Blues

- Website: Generate Music Using TensorFlow and Python
  https://rubikscode.net/2018/11/12/generate-music-using-tensorflow-and-python/
"""
import conversions
import numpy as np
import glob
from tqdm import tqdm
import os
from tensorflow.python.ops import control_flow_ops
import tensorflow.compat.v1 as tf
import shutil
tf.disable_v2_behavior()
# This function takes probabilities probability as input and returns a rounded binary sample based on the probabilities
def sample(probability):
    return tf.floor(probability + tf.random.uniform(tf.shape(probability), 0, 1, dtype=tf.float32))

# This function represents a single step of the Gibbs sampling algorithm 
# In each step, it calculates the hidden state hk by applying the sigmoid activation function to the product of xk and the weight matrix, added with the hidden bias. 
def gibbs_step(count, k, xk):
    hk = sample(tf.sigmoid(tf.matmul(xk, weight_matrix) + hidden_bias))
    xk = sample(tf.sigmoid(tf.matmul(hk, tf.transpose(weight_matrix)) + visible_bias))
    return count + 1, k, xk
# This function performs the Gibbs sampling procedure for k times. It initializes a counter to 0 and then uses a while loop to repeatedly call the gibbs_step function
def gibbs_sample(k):
    counter = tf.constant(0)
    [_, _, x_sample] = tf.while_loop(
        lambda count, num_iter, *args: count < num_iter,
        gibbs_step,
        [counter, tf.constant(k), x],
        parallel_iterations=1,
        back_prop=True,  # Set back_prop=True to enable gradients to flow through the loop
        swap_memory=False
    )
    # Use tf.stop_gradient to stop gradients from flowing through x_sample
    x_sample = tf.stop_gradient(x_sample)
    return x_sample

# Set numpy display options to show entire array
np.set_printoptions(threshold=np.inf, edgeitems=None)

data_path = "../midis/"
# Search for all files in the directory specified by the data_path variable that have a .mid or .midi extension
file_list = glob.glob('{}/*.*mid*'.format(data_path))
# This is getting all of the input midis and putting them in a list
# If a midi file is not valid it will skip over that one and keep going
song_list = []
for files in tqdm(file_list):
    try:
        song = np.array(conversions.midi_conversion(files))
        if np.array(song).shape[0] > 50:
            song_list.append(song)
    except Exception as exc:
        print("Error occurred while processing {}: {}".format(files, exc))
print("Processed {} valid files out of {} total files.".format(len(song_list), len(file_list)))
# This is setting the boundary for how wide our key selection can be which is defined in the conversions folder
lowest = conversions.lower_note_bound
highest = conversions.upper_note_bound
range_notes = highest - lowest

steps = 2000  # Number of notes to create at a time
visible_size = 2 * range_notes * steps  # Visible layer size
hidden_size = 10  # Hidden layer size
epochs = 100 # Number of opechs
batch = 1 # batch size
learning_rate = tf.constant(1, tf.float32)  # Learning rate

# Placeholders and variables for the network

x = tf.placeholder(tf.float32, [None, visible_size], name="x")  # Placeholder for data
weight_matrix = tf.Variable(tf.random_normal([visible_size, hidden_size], 0.01), name="weight_matrix")  # Weight matrix
hidden_bias = tf.Variable(tf.zeros([1, hidden_size], tf.float32, name="hidden_bias"))  # Bias for hidden layer
visible_bias = tf.Variable(tf.zeros([1, visible_size], tf.float32, name="visible_bias"))  # Bias for visible layer

#  The gibbs_sample function performs Gibbs sampling, which is a technique used to approximate the distribution of a probabilistic model. 
# In this case, it is used to approximate the distribution of an RBM
x_sample = gibbs_sample(5)

# This line computes the hidden units' activations h using the visible units' activations x, weight matrix, and hidden bias 
h = sample(tf.sigmoid(tf.matmul(x, weight_matrix) + hidden_bias))

# This line computes a sample h_sample of the hidden units' activations using the x_sample obtained from the Gibbs sampling step.
h_sample = sample(tf.sigmoid(tf.matmul(x_sample, weight_matrix) + hidden_bias))

# This line calculates the size of the training data x
training_data_size = tf.cast(tf.shape(x)[0], tf.float32)

#  This line calculates the learning rate per training example by dividing the learning_rate (a constant) by the training_data_size
# The learning rate step_size is the step size used for updating the RBM's parameters.
step_size = learning_rate / training_data_size

# This line calculates the update to the weight matrix
weight_matrix_update = tf.multiply(step_size, tf.subtract(tf.matmul(tf.transpose(x), h), tf.matmul(tf.transpose(x_sample), h_sample)))

# Visibal Bias update
visible_bias_update = tf.multiply(step_size, tf.reduce_sum(tf.subtract(x, x_sample), axis=0, keepdims=True))

# Hidden bias update
hidden_bias_update = tf.multiply(step_size, tf.reduce_sum(tf.subtract(h, h_sample), axis=0, keepdims=True))

# This line combines the updates for weight_matrix, visible_bias, and hidden_bias into a list called upd_list
upd_list = [weight_matrix.assign_add(weight_matrix_update), visible_bias.assign_add(visible_bias_update), hidden_bias.assign_add(hidden_bias_update)]

# The session is responsible for executing TensorFlow operations and running computational graphs.
TFsess = tf.Session()

# initializes all the variables in the current TensorFlow graph
init = tf.initialize_all_variables()
TFsess.run(init)

for epoch in tqdm(range(epochs)):
    for song in song_list:
        song = np.array(song)
        # Calculates the number of blocks that can be created from the song array based on the specified steps
        blocks = int(song.shape[0] / steps)
        # Calculates the maximum duration possible based on the number of blocks and steps
        duration = int(blocks * steps)
        song = song[:duration]
        #  Reshapes the song array into blocks of size steps by flattening each block into a 1-dimensional array.
        song = np.reshape(song, [blocks, song.shape[1] * steps])
        for i in range(1, len(song), batch):
            training_examples = song[i:i + batch]
            TFsess.run(upd_list, feed_dict={x: training_examples})

#  generates a sample from the RBM model by running the gibbs_sample operation within the TensorFlow session
sample = gibbs_sample(1).eval(session=TFsess, feed_dict={x: np.zeros((1, visible_size))})
# Reshapes the generated sample from a 1-dimensional array into a 2-dimensional array
reshape_sample = np.reshape(sample[0, :], (steps, 2 * range_notes))
# Converts the reshaped sample array into a MIDI-compatible format
conversions.state_matrix_conversion(reshape_sample, "new_song")

