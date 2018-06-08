#!/usr/bin/python3

import json
import sys
import itertools
from multiprocessing import Pool

PREF='../logs/log'
SUF='.log'
THREADS=20
#~ THREADS=2


def process_file(filename):
	turns = []
	count = 0
	try:
		j = json.load(open(filename, 'r'))
		games = j['game']
		for game in games:
			action = game[1]['player_move']
			if action == "End turn":
				turns.append(count)
				count = 0
			else:
				move = action.split()
				if move[0] == 'Play:':
					count += 1
		turns.append(count)
	except:
		error("Parsing {} failed".format(filename))
	return turns

def reduce(outs):
	turns = list(itertools.chain.from_iterable(outs))
	non_zero = [e for e in turns if e != 0]
	print("Niezerowe tury średnio: {}".format(sum(non_zero) / float(len(non_zero))))
	return turns



def error(msg):
	print(msg, file=sys.stderr)


if __name__ == "__main__":
	if len(sys.argv) == 4:
		pool = Pool(THREADS)
		logs = ["{}{}{}".format(PREF, num, SUF) for num in range(int(sys.argv[1]), int(sys.argv[2]))]
		outs = pool.map(process_file, logs)
		out = reduce(outs)
		with open(sys.argv[3], 'w') as f:
			print(*out, sep="\n", file=f)
	else:
		error("Usage {} range_from range_to output_file".format(sys.argv[0]))
