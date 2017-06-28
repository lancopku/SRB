from __future__ import print_function
import sys
import os
import tensorflow as tf
import time
from Unit import *
from DataLoader import DataLoader
import numpy as np
from ROUGE.PythonROUGE import PythonROUGE

tf.app.flags.DEFINE_string("cell", "lstm", "Rnn cell.")
tf.app.flags.DEFINE_integer("hidden_size", 500, "Size of each layer.")
tf.app.flags.DEFINE_integer("emb_size", 400, "Size of embedding.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size of train set.")
tf.app.flags.DEFINE_integer("epoch", 50, "Number of training epoch.")
tf.app.flags.DEFINE_float("dropout", 1.0, "Dropout keep probability.")
tf.app.flags.DEFINE_string("gpu", '0', "GPU id.")
tf.app.flags.DEFINE_string("opt",'Adam','Optimizer.')
tf.app.flags.DEFINE_string("mode",'train','train or test')
tf.app.flags.DEFINE_string("save",'0','save directory')
tf.app.flags.DEFINE_string("load",'0','load directory')
tf.app.flags.DEFINE_string("dir",'lcsts','data set directory')
tf.app.flags.DEFINE_integer("limits",0,'max data set size')
tf.app.flags.DEFINE_boolean("ckpt", False,'load checkpoint or not')
tf.app.flags.DEFINE_boolean("attention", True,'attention mechanism or not')
tf.app.flags.DEFINE_boolean("dev", False,'dev or test')
tf.app.flags.DEFINE_boolean("SRB", True,'use SRB or test')
tf.app.flags.DEFINE_integer("source_vocab", 4003,'vocabulary size')
tf.app.flags.DEFINE_integer("target_vocab", 4003,'vocabulary size')
tf.app.flags.DEFINE_integer("report", 500,'report')
tf.app.flags.DEFINE_string("m","train",'running message')
FLAGS = tf.app.flags.FLAGS

gold_path = FLAGS.dir + '/evaluation/test_gold_summarys_'
pred_path = FLAGS.dir + '/evaluation/test_pred_summarys_'

if FLAGS.save == "0":
    save_dir = FLAGS.dir + '/' + str(int(time.time() * 1000)) + '/'
    os.mkdir(save_dir)
else:
    save_dir = FLAGS.save
log_file = save_dir + 'log.txt'

def train(sess, dataloader, model):
    write_log("#######################################################\n")
    for flag in FLAGS.__flags:
        write_log(flag + " = " + str(FLAGS.__flags[flag]) + "\n")
    write_log("#######################################################\n")
    trainset = dataloader.train_set
    k = 0
    for _ in range(FLAGS.epoch):
        loss, start_time = 0.0, time.time()
        for x in dataloader.batch_iter(trainset, FLAGS.batch_size, True):
            loss += model(x, sess)
            k += 1
            sys.stdout.write('training %.2f ...\r' % (k % FLAGS.report * 100.0 / FLAGS.report))
            sys.stdout.flush()
            if (k % FLAGS.report == 0):
                cost_time = time.time() - start_time
                #print("%d : loss = %.3f, time = %.3f" % (k // FLAGS.report, loss, cost_time), end=' ')
                write_log("%d : loss = %.3f, time = %.3f " % (k // FLAGS.report, loss, cost_time))
                loss, start_time = 0.0, time.time()
                write_log(evaluate(sess, dataloader, model))
                model.save(save_dir)


def test(sess, dataloader, model):
    model.load(save_dir)
    print(evaluate(sess, dataloader, model, FLAGS.dev), end='')


def evaluate(sess, dataloader, model, dev=False):
    if dev:
        evalset = dataloader.dev_set
    else:
        evalset = dataloader.test_set
    k = 0
    for x in dataloader.batch_iter(evalset, FLAGS.batch_size, False):
        predictions = model.generate(x, sess)
        for summary in np.array(predictions):
            with open(pred_path + str(k), 'w') as sw:
                summary = list(summary)
                if 2 in summary:
                    summary = summary[:summary.index(2)] if summary[0] != 2 else [2]
                sw.write(" ".join([str(x) for x in summary]) + '\n')
                k += 1
    # print(k)
    pred_set = [pred_path + str(i) for i in range(k)]
    gold_set = [[gold_path + str(i)] for i in range(k)]
    recall, precision, F_measure = PythonROUGE(pred_set, gold_set, ngram_order=2)
    result = "F_measure: %s Recall: %s Precision: %s\n" % (str(F_measure), str(recall), str(precision))
    #print(result)
    return result

def write_log(s):
    print(s, end='')
    with open(log_file, 'a') as f:
        f.write(s)

def main():
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        dataloader = DataLoader(FLAGS.dir, FLAGS.limits)
        model = SeqUnit(batch_size=FLAGS.batch_size, hidden_size=FLAGS.hidden_size, emb_size=FLAGS.emb_size,
                        source_vocab=FLAGS.source_vocab, target_vocab=FLAGS.target_vocab, scope_name="seq2seq",
                        name="seq2seq", attention=FLAGS.attention, SRB=FLAGS.SRB)
        sess.run(tf.global_variables_initializer())
        if FLAGS.load != '0':
            model.load(FLAGS.load)
        if FLAGS.mode == 'train':
            train(sess, dataloader, model)
        else:
            test(sess, dataloader, model)

if __name__=='__main__':
    with tf.device('/gpu:' + FLAGS.gpu):
        main()