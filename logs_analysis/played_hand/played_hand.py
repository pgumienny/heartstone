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
	cards = []
	try:
		j = json.load(open(filename, 'r'))
		games = j['game']
		for game in games:
			move = game[1]['player_move'].split()
			if move[0] == 'Play:':
				played = '_'.join(move[3:])
				for card in game[0]['player']['hand']:
					name = card['name'].replace(" ", "_")
					if name != played:
						cards.append((played, name))
	except:
		error("Parsing {} failed".format(filename))
	return cards

def reduce(outs):
	cards = [ "{} {}".format(card, context) for (card, context) in list(itertools.chain.from_iterable(outs))]
	return cards



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
