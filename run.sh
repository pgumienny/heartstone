#!/bin/bash

# rm -rf log_tb && (tensorboard --logdir=log_tb &) && python word2vec_basic.py

for file in samples/*
do
	short_file=$(basename $file)
	python word2vec_basic.py --log_dir="logs/$short_file" --input_file="$file"
done



