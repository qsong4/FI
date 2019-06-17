import tensorflow as tf
import math
import os
from model import FI
from data_load import get_batch_infer
from hparams import Hparams
import time

class Inference(object):
    def __init__(self):
        '''
        Here only load params from ckpt, and change read input method from dataset without placehold
        to dataset with placeholder. Because withuot placeholder you cannt init model when class build which
        means you spend more time on inference stage.
        '''
        hparams = Hparams()
        parser = hparams.parser

        self.hp = parser.parse_args()
        self.m = FI(self.hp)
        self.pred_op = self.m.predict_model()

        self.sess = tf.Session()
        ckpt = tf.train.latest_checkpoint(self.hp.modeldir)
        saver = tf.train.Saver()
        saver.restore(self.sess, ckpt)


    def infer(self, sents1, sents2):
        infer_batches = get_batch_infer(sents1, sents2, self.hp.vocab, self.hp.batch_size)
        iter = tf.data.Iterator.from_structure(infer_batches.output_types, infer_batches.output_shapes)
        infer_init_op = iter.make_initializer(infer_batches)
        self.sess.run(infer_init_op)
        data_element = iter.get_next()
        x, y = self.sess.run(data_element)

        feed_dict = self.m.create_feed_dict_infer(x, y)
        pred_res = self.sess.run(self.pred_op, feed_dict=feed_dict)

        return pred_res

if __name__ == '__main__':
    sents1 = ["今天天气不错"]
    sents2 = ["我们都很开心"]
    inf = Inference()
    for i in range(2):
        start = time.time()
        res = inf.infer(sents1, sents2)
        end = time.time()
        print(end - start)
    print(res)


