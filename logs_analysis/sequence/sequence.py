#!/usr/bin/python3

import json
import sys
import itertools
from multiprocessing import Pool

PREF='../logs/log'
SUF='.log'
THREADS=20
#~ THREADS=2

WIN=4

def process_file(filename):
	cards = []
	try:
		j = json.load(open(filename, 'r'))
		games = j['game']
		played = [[], []]
		for game in games:
			action = game[1]['player_move']
			if action == "End turn":
				played[game[0]['active_player']].append("End")
			else:
				move = action.split()
				if move[0] == 'Play:':
					played[game[0]['active_player']].append('_'.join(move[3:]))
		for p in [0, 1]:
			for i in range(0, max(0, len(played[p]) - WIN + 1)):
				if (played[p][i] != "End"):
					cards.append(played[p][i:i+WIN])
	except:
		error("Parsing {} failed".format(filename))
	return cards

def reduce(outs):
	cards = [ " ".join(end) for end in list(itertools.chain.from_iterable(outs))]
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
