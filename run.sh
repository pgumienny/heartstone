#!/bash/bin

rm -rf log_tb && (tensorboard --logdir=log_tb &) && python word2vec_basic.py
