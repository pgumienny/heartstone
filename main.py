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


def process_file(filename):
	# print(filename)
	filename = "logs/" + filename
	moves = get_moves_lists(filename)
	if moves:
		return moves[0] + ["."] + moves[1] + ["."]
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
						move = ' '.join(move[3:])
						# print("Play(" + str(active_player) + ") = " + move)
						move_list[active_player].append(move)
		return move_list
	except:
		return [[],[]]
	# print(move_list)

pool = ThreadPool(8)
text = pool.map(process_file, os.listdir("logs/"))
text = list(itertools.chain.from_iterable(text))
print ("\n".join(text))
word_to_index = defaultdict(count(1).__next__)

###
#
#   TU SIE ZACZYNA KOD TENSORFLOW, JESZCZE NIE DZIALA
#
###


# word_ids = []
# for word in text:
#     word_ids.append(word_to_index[word])

# vocabulary_size = max(word_ids) + 1
# embedding_size = 40
# batch_size = 32

# print("text calculated", file=sys.stdout)

# with tf.Session() as sess:
# 	embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
# 	nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
# 	nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
# 	train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
# 	train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
# 	embed = tf.nn.embedding_lookup(embeddings, train_inputs)
# 	loss = tf.reduce_mean(
# 				  tf.nn.nce_loss(weights=nce_weights,
# 				                 biases=nce_biases,
# 				                 labels=train_labels,
# 				                 inputs=embed,
# 				                 num_sampled=num_sampled,
# 				                 num_classes=vocabulary_size))


# 	init_op = tf.global_variables_initializer()
# 	sess.run(init_op)
# 	print(sess.run(embedded_word_ids)) #execute init_op

# print(word_embeddings)
