import tensorflow as tf

from data_load import load_vocab
from modules import get_token_embeddings, ff, positional_encoding, multihead_attention, label_smoothing, noam_scheme
import logging

logging.basicConfig(level=logging.INFO)


class FI:
    """
    xs: tuple of
        x: int32 tensor. (句子长度，)
        x_seqlens. int32 tensor. (句子)
    """

    def __init__(self, hp):
        self.hp = hp
        self.token2idx, self.idx2token, self.hp.vocab_size = load_vocab(hp.vocab)
        self.embeddings = get_token_embeddings(self.hp.vocab_size, self.hp.d_model, zero_pad=True)

    def representation(self, xs, ys, training=True):
        with tf.variable_scope("representation", reuse=tf.AUTO_REUSE):
            x, x_seqlen = xs
            y, y_seqlen = ys

            #print(x)
            #print(y)

            # embedding
            encx = tf.nn.embedding_lookup(self.embeddings, x)  # (N, T1, d_model)
            encx *= self.hp.d_model ** 0.5  # scale

            encx += positional_encoding(encx, self.hp.maxlen)
            encx = tf.layers.dropout(encx, self.hp.dropout_rate, training=training)

            ency = tf.nn.embedding_lookup(self.embeddings, y)  # (N, T1, d_model)
            ency *= self.hp.d_model ** 0.5  # scale

            ency += positional_encoding(ency, self.hp.maxlen)
            ency = tf.layers.dropout(ency, self.hp.dropout_rate, training=training)

            ##TODO:add abcnn first layer attention

            ## Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    encx = multihead_attention(queries=encx,
                                               keys=encx,
                                               values=encx,
                                               num_heads=self.hp.num_heads,
                                               dropout_rate=self.hp.dropout_rate,
                                               training=training,
                                               causality=False)
                    # feed forward
                    encx = ff(encx, num_units=[self.hp.d_ff, self.hp.d_model])

                    ency = multihead_attention(queries=ency,
                                               keys=ency,
                                               values=ency,
                                               num_heads=self.hp.num_heads,
                                               dropout_rate=self.hp.dropout_rate,
                                               training=training,
                                               causality=False)

                    # feed forward
                    ency = ff(ency, num_units=[self.hp.d_ff, self.hp.d_model])
        return encx, ency

    def interactivate(self, a_repre, b_repre, training=True):
        with tf.variable_scope("interactivate", reuse=tf.AUTO_REUSE):
            # Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # Vanilla attention
                    dec = multihead_attention(queries=b_repre,
                                              keys=a_repre,
                                              values=a_repre,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False,
                                              scope="vanilla_attention")
                    ### Feed Forward
                    dec = ff(dec, num_units=[self.hp.d_ff, self.hp.d_model])
        return dec

    def fc(self, inpt, match_dim):
        with tf.variable_scope("fc", reuse=tf.AUTO_REUSE):
            w = tf.get_variable("w", [match_dim, self.hp.num_class], dtype=tf.float32)
            b = tf.get_variable("b", [self.hp.num_class], dtype=tf.float32)
            logits = tf.matmul(inpt, w) + b
        # prob = tf.nn.softmax(logits)

        # gold_matrix = tf.one_hot(labels, self.hp.num_class, dtype=tf.float32)
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
        return logits

    # calculate classification accuracy
    def _acc_op(self):
        with tf.name_scope('acc'):
            label_pred = tf.argmax(self.logits, 1, name='label_pred')
            label_true = tf.argmax(self.y, 1, name='label_true')
            correct_pred = tf.equal(tf.cast(label_pred, tf.int32), tf.cast(label_true, tf.int32))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='Accuracy')
        return accuracy

    def train(self, xs, ys, labels):
        # representation
        x_repre, y_repre = self.representation(xs, ys)
        #print(labels.shape)
        # interactivate
        x_inter = self.interactivate(x_repre, y_repre)  # (?, ?, 512)
        y_inter = self.interactivate(y_repre, x_repre)  # (?, ?, 512)

        x_avg = tf.reduce_mean(x_inter, axis=1)
        y_avg = tf.reduce_mean(y_inter, axis=1)
        x_max = tf.reduce_max(x_inter, axis=1)
        y_max = tf.reduce_max(y_inter, axis=1)

        input2fc = tf.concat([x_avg, x_max, y_avg, y_max], axis=1)
        #input2fc = tf.concat([x_inter, y_inter], 2)  # (?, ?, 1024)
        #print(input2fc.shape.as_list())
        logits = self.fc(input2fc, match_dim=input2fc.shape.as_list()[-1])
        #print(logits.shape)


        #gold_matrix = tf.one_hot(labels, self.hp.num_class, dtype=tf.float32)
        #aaa = tf.print(logits, ["LOGITS", logits])
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
        with tf.variable_scope('acc', reuse=tf.AUTO_REUSE):
            label_pred = tf.argmax(logits, 1, name='label_pred')
            label_true = tf.argmax(labels, 1, name='label_true')
            correct_pred = tf.equal(tf.cast(label_pred, tf.int32), tf.cast(label_true, tf.int32))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='Accuracy')

        # train scheme
        global_step = tf.train.get_or_create_global_step()
        #lr = noam_scheme(self.hp.lr, global_step, self.hp.warmup_steps)
        optimizer = tf.train.AdamOptimizer(self.hp.lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

        return loss, train_op, global_step, accuracy

    def eval(self, xs, ys, labels):
        # representation
        x_repre, y_repre = self.representation(xs, ys)
        # y_repre = self.representation(ys)

        # interactivate
        x_inter = self.interactivate(x_repre, y_repre)  # (?, ?, 512)
        y_inter = self.interactivate(y_repre, x_repre)  # (?, ?, 512)
        # print(y_inter.shape)
        # print(x_inter.shape)

        x_avg = tf.reduce_mean(x_inter, axis=1)
        y_avg = tf.reduce_mean(y_inter, axis=1)
        x_max = tf.reduce_max(x_inter, axis=1)
        y_max = tf.reduce_max(y_inter, axis=1)

        input2fc = tf.concat([x_avg, x_max, y_avg, y_max], axis=1)

        logits = self.fc(input2fc, match_dim=input2fc.shape.as_list()[-1])

        #gold_matrix = tf.one_hot(labels, self.hp.num_class, dtype=tf.float32)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))

        with tf.variable_scope('acc', reuse=tf.AUTO_REUSE):
            label_pred = tf.argmax(logits, 1, name='label_pred')
            label_true = tf.argmax(labels, 1, name='label_true')
            correct_pred = tf.equal(tf.cast(label_pred, tf.int32), tf.cast(label_true, tf.int32))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='Accuracy')

        return accuracy, loss
