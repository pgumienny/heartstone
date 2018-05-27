import tensorflow as tf
import json
import sys
import os
import math
import functools
from itertools import count
import itertools
from collections import defaultdict
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np


import collections
import argparse
import random
from tempfile import gettempdir
import zipfile

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin


batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 2  # How many words to consider left and right.
num_skips = 4  # How many times to reuse an input to generate a label.
num_sampled = 64  # Number of negative examples to sample.
num_step = 10000000 # number of DL steps
LOGDIR = "biglog/" # Log directory

def process_file(filename):
	# print(filename)
	filename = LOGDIR + filename
	moves = get_moves_lists(filename)
	if moves:
		# return moves[0] + ["game_end_end"] + moves[1] + ["."]
		return moves[0] + ["game_end_end"]
	else:
		return []


def get_moves_lists(filename):
	# print(filename)
	try:
		j = json.load(open(filename, 'r'))
		move_list = [[], []]
		for i, game in j.items():
			if i == "game":
				for step in game:
					active_player = step[0]['active_player']
					move = step[1]['player_move'].split()
					if move[0] == "Play:":
						move = ' '.join(move[3:]).replace(' ', '_')
						# print("Play(" + str(active_player) + ") = " + move)
						# move_list[active_player].append(move)
						move_list[0].append(move)
		return move_list
	except:
		return [[],[]]
	# print(move_list)

pool = ThreadPool(8)
text = pool.map(process_file, os.listdir(LOGDIR))

# text = ""
# with open('text.txt', 'r') as myfile:
#     text=myfile.read().replace('\n', ' ').split()



text = list(itertools.chain.from_iterable(text))
print(" ".join(text))


word_to_index = defaultdict(count(1).__next__)

###
#
#   TU SIE ZACZYNA KOD TENSORFLOW, JESZCZE NIE DZIALA
#
###


word_ids = []
for word in text:
	word_ids.append(word_to_index[word])

# print(word_to_index, file=sys.stderr)
# print(' '.join(str(x) for x in word_ids))
# print(" ".join(word_ids))

vocabulary_size = max(word_ids) + 1

print("text calculated", file=sys.stderr)
print("vocabulary_size = ", vocabulary_size, file=sys.stderr)
print("text_length = ", len(text), file=sys.stderr)

sys.exit()

data = word_ids
data_index = 0
# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
	global data_index
	assert batch_size % num_skips == 0
	assert num_skips <= 2 * skip_window
	batch = np.ndarray(shape=(batch_size), dtype=np.int32)
	labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
	span = 2 * skip_window + 1  # [ skip_window target skip_window ]
	buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin
	if data_index + span > len(data):
		data_index = 0
	buffer.extend(data[data_index:data_index + span])
	data_index += span
	for i in range(batch_size // num_skips):
		context_words = [w for w in range(span) if w != skip_window]
		words_to_use = random.sample(context_words, num_skips)
		for j, context_word in enumerate(words_to_use):
			batch[i * num_skips + j] = buffer[skip_window]
			labels[i * num_skips + j, 0] = buffer[context_word]
		if data_index == len(data):
			buffer.extend(data[0:span])
			data_index = span
		else:
			buffer.append(data[data_index])
			data_index += 1
	# Backtrack a little bit to avoid skipping words in the end of a batch
	data_index = (data_index + len(data) - span) % len(data)
	return batch, labels


with tf.Session() as sess:
	embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
	nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
	nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
	train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
	train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
	embed = tf.nn.embedding_lookup(embeddings, train_inputs)
	loss = tf.reduce_mean(
				  tf.nn.nce_loss(weights=nce_weights,
								 biases=nce_biases,
								 labels=train_labels,
								 inputs=embed,
								 num_sampled=num_sampled,
								 num_classes=vocabulary_size))


	init_op = tf.global_variables_initializer()
	sess.run(init_op)

	train_step = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)

	average_loss = 0
	for step in range(num_step):

		batch_inputs, batch_labels = generate_batch(batch_size, num_skips,
													skip_window)
		feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
		_, loss_val = sess.run(
			[train_step, loss],
			feed_dict=feed_dict)
		average_loss += loss_val

		if step % 1000 == 0:
			if step > 0:
				average_loss /= 1000
				# The average loss is an estimate of the loss over the last 1000 batches.
				print(step, ', ', average_loss)
				sys.stdout.flush()
				average_loss = 0


# print(word_embeddings)
