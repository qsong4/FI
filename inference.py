# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = -1

import tensorflow as tf
from model import FI
from data_load import get_batch_infer, get_batch
from hparams import Hparams
from tqdm import tqdm

class Inference(object):
    def __init__(self, is_test=False):
        '''
        Here only load params from ckpt, and change read input method from dataset without placehold
        to dataset with placeholder. Because withuot placeholder you cannt init model when class build which
        means you spend more time on inference stage.
        '''
        hparams = Hparams()
        parser = hparams.parser

        self.hp = parser.parse_args()
        self.m = FI(self.hp)
        if not is_test:
            self.pred_op = self.m.predict_model()
        else:
            self.loss_op, self.acc_op = self.m.eval_model()


        self.sess = tf.Session()
        ckpt = tf.train.latest_checkpoint(self.hp.modeldir)
        saver = tf.train.Saver()
        saver.restore(self.sess, ckpt)

    def test_file(self):
        test_file = self.hp.test_file
        test_batches, num_test_batches, num_test_samples = get_batch(test_file, self.hp.maxlen,
                                                                        self.hp.vocab, self.hp.batch_size)
        iter = tf.data.Iterator.from_structure(test_batches.output_types, test_batches.output_shapes)
        data_element = iter.get_next()
        test_init_op = iter.make_initializer(test_batches)

        self.sess.run(test_init_op)
        x, y, x_len, y_len, labels = self.sess.run(data_element)
        feed_dict = self.m.create_feed_dict(x, y, x_len, y_len, labels)
        total_steps = 1 * num_test_batches
        total_acc = 0.0
        total_loss = 0.0
        for i in tqdm(range(total_steps + 1)):
            # dev_acc, dev_loss = sess.run([dev_accuracy_op, dev_loss_op])
            test_acc, test_loss = self.sess.run([self.acc_op, self.loss_op], feed_dict=feed_dict)
            total_acc += test_acc
            total_loss += test_loss

        return total_acc / total_steps


    def infer(self, sents1, sents2):
        infer_batches = get_batch_infer(sents1, sents2, self.hp.maxlen, self.hp.vocab, self.hp.batch_size)
        iter = tf.data.Iterator.from_structure(infer_batches.output_types, infer_batches.output_shapes)
        infer_init_op = iter.make_initializer(infer_batches)
        self.sess.run(infer_init_op)
        data_element = iter.get_next()
        x, y, x_len, y_len = self.sess.run(data_element)

        feed_dict = self.m.create_feed_dict_infer(x, y, x_len, y_len)
        pred_res = self.sess.run(self.pred_op, feed_dict=feed_dict)

        return pred_res

if __name__ == '__main__':
    inf = Inference(is_test=True)
    """
    sents1 = ["今天天气不错"]
    sents2 = ["我们都很开心"]
    
    for i in range(2):
        start = time.time()
        res = inf.infer(sents1, sents2)
        end = time.time()
        print(end - start)
    print(res)
    """
    acc = inf.test_file()


