#!/usr/bin/python3

import json
import sys
import itertools
from multiprocessing import Pool
import operator

PREF='../logs/log'
SUF='.log'
THREADS=20
#~ THREADS=2

WIN=3

def process_file(filename):
	card = None
	winner = None
	try:
		j = json.load(open(filename, 'r'))
		game = j['game'][2]
		action = game[1]['player_move']
		if action == "End turn":
			card = "End"
		else:
			move = action.split()
			if move[0] == 'Play:':
				card = '_'.join(move[3:])
		player = game[0]['active_player']
		winner = int(j['result'].split()[0].split("_")[1])
	except:
		error("Parsing {} failed: {}".format(filename, sys.exc_info()))
	return [(card, winner)]

def reduce(outs):
	d = {}
	for (card, winner) in list(itertools.chain.from_iterable(outs)):
		if not card in d:
			d[card] = [0, 0]
		d[card][winner] += 1
	l = []
	for (card, (win0, win1)) in d.items():
		s = win0 + win1
		p = max(win0/s, win1/s)
		l.append((card, p))
	l.sort(key = operator.itemgetter(1))
	l = [ "{} {}".format(card, perc) for (card, perc) in l]
	return l



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
