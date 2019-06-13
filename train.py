# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
import tensorflow as tf

from model import FI
from tqdm import tqdm
from data_load import get_batch
from utils import save_hparams, save_variable_specs, get_hypotheses, calc_bleu
import os
from hparams import Hparams
import math

print("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()

print("# Prepare train/eval batches")
train_batches, num_train_batches, num_train_samples = get_batch(hp.train, hp.maxlen,
                                                                hp.vocab, hp.batch_size, shuffle=True)
eval_batches, num_eval_batches, num_eval_samples = get_batch(hp.eval, hp.maxlen,
                                                             hp.vocab, hp.batch_size,shuffle=False)

# create a iterator of the correct shape and type
iter = tf.data.Iterator.from_structure(train_batches.output_types, train_batches.output_shapes)
xs, ys, labels= iter.get_next()

train_init_op = iter.make_initializer(train_batches)
eval_init_op = iter.make_initializer(eval_batches)

print("# Load model")
m = FI(hp)
loss_op, train_op, global_step, accuracy_op = m.train(xs, ys, labels)
dev_accuracy_op, dev_loss_op = m.eval(xs, ys, labels)
# y_hat = m.infer(xs, ys)

print("# Session")
saver = tf.train.Saver(max_to_keep=hp.num_epochs)
with tf.Session() as sess:
    ckpt = tf.train.latest_checkpoint(hp.modeldir)
    if ckpt is None:
        print("Initializing from scratch")
        sess.run(tf.global_variables_initializer())
        save_variable_specs(os.path.join(hp.modeldir, "specs"))
    else:
        saver.restore(sess, ckpt)


    sess.run(train_init_op)
    total_steps = hp.num_epochs * num_train_batches
    _gs = sess.run(global_step)
    best_acc = 0
    for i in tqdm(range(_gs, total_steps+1)):
        _, _gs = sess.run([train_op, global_step])
        epoch = math.ceil(_gs / num_train_batches)

        if _gs and _gs % num_train_batches == 0:
            print("\n")
            print("epoch {} is done".format(epoch))
            _loss, _accuracy = sess.run([loss_op, accuracy_op]) # train loss and accuracy
            print("# train results")
            print("训练集: loss {:g}, acc {:g} \n".format(dev_loss, dev_acc))
            print("# test evaluation")
            _, dev_acc, dev_loss = sess.run([eval_init_op, dev_accuracy_op, dev_loss_op])

            print("# evaluation results")
            print("验证集: loss {:g}, acc {:g} \n".format(dev_loss, dev_acc))
            if dev_acc > best_acc:
                best_acc = dev_acc
                print("# save models")
                model_output = hp.model_path % (epoch, dev_loss, dev_acc)
                ckpt_name = os.path.join(hp.modeldir, model_output)
                saver.save(sess, ckpt_name, global_step=_gs)
                print("after training of {} epochs, {} has been saved.".format(epoch, ckpt_name))

            print("# fall back to train mode")
            sess.run(train_init_op)


print("Done")
