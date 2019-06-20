import tensorflow as tf

from data_load import load_vocab, loadGloVe
from modules import get_token_embeddings, ff, positional_encoding, multihead_attention, label_smoothing, noam_scheme


class FI:
    """
    xs: tuple of
        x: int32 tensor. (句子长度，)
        x_seqlens. int32 tensor. (句子)
    """

    def __init__(self, hp):
        self.hp = hp
        self.token2idx, self.idx2token, self.hp.vocab_size = load_vocab(hp.vocab)
        self.embd = None
        if self.hp.preembedding:

            self.embd = loadGloVe(self.hp.vec_path)
        self.embeddings = get_token_embeddings(self.embd, self.hp.vocab_size, self.hp.d_model, zero_pad=False)
        self.x = tf.placeholder(tf.int32, [None, None], name="text_x")
        self.y = tf.placeholder(tf.int32, [None, None], name="text_y")
        self.truth = tf.placeholder(tf.int32, [None, self.hp.num_class], name="truth")

    def create_feed_dict(self, x, y, truth):
        feed_dict = {
            self.x: x,
            self.y: y,
            self.truth: truth,
        }

        return feed_dict

    def create_feed_dict_infer(self, x, y):
        feed_dict = {
            self.x: x,
            self.y: y
        }

        return feed_dict

    def make_attention_mat(self, x1, x2):
        euclidean = tf.sqrt(tf.reduce_sum(tf.square(x1 - tf.matrix_transpose(x2)), axis=1) + 1e-8)
        return 1 / (1 + euclidean)

    def representation(self, xs, ys, training=True):
        with tf.variable_scope("representation", reuse=tf.AUTO_REUSE):
            x = xs
            y = ys

            # print(x)
            # print(y)

            # embedding
            encx = tf.nn.embedding_lookup(self.embeddings, x)  # (N, T1, d_model)
            encx *= self.hp.d_model ** 0.5  # scale

            encx += positional_encoding(encx, self.hp.maxlen)
            encx = tf.layers.dropout(encx, self.hp.dropout_rate, training=training)

            ency = tf.nn.embedding_lookup(self.embeddings, y)  # (N, T1, d_model)
            ency *= self.hp.d_model ** 0.5  # scale

            ency += positional_encoding(ency, self.hp.maxlen)
            ency = tf.layers.dropout(ency, self.hp.dropout_rate, training=training)

            # encx = tf.transpose(encx, [0, 2, 1])
            # ency = tf.transpose(ency, [0, 2, 1])
            # encx_expand = tf.expand_dims(encx, -1)
            # ency_expand = tf.expand_dims(ency, -1)
            #
            # ##TODO:add abcnn first layer attention
            # aW = tf.get_variable(name="aW", shape=(self.hp.maxlen, self.hp.d_model),
            #                      initializer=tf.contrib.layers.xavier_initializer(),
            #                      regularizer=tf.contrib.layers.l2_regularizer(scale=0.001))
            # att_mat = self.make_attention_mat(encx_expand, ency_expand)
            # #print(att_mat.shape)
            # encx_a = tf.einsum("ijk,kl->ijl", att_mat, aW)
            # ency_a = tf.einsum("ijk,kl->ijl", tf.matrix_transpose(att_mat), aW)
            #
            # encx = tf.transpose(encx, [0, 2, 1])
            # ency = tf.transpose(ency, [0, 2, 1])
            # #print(encx.shape)
            # #print(encx_a.shape)
            #
            # # encx = tf.concat([encx, encx_a], axis=2)
            # # ency = tf.concat([ency, ency_a], axis=2)
            # encx += encx_a
            # ency += ency_a

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

    def fc(self, inpt, match_dim, reuse=tf.AUTO_REUSE):
        with tf.variable_scope("fc", reuse=reuse):
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

    def build_model(self):
        # representation
        x_repre, y_repre = self.representation(self.x, self.y)
        # y_repre = self.representation(self.y)
        # print(labels.shape)
        # interactivate
        x_inter = self.interactivate(x_repre, y_repre)  # (?, ?, 512)
        y_inter = self.interactivate(y_repre, x_repre)  # (?, ?, 512)

        x_avg = tf.reduce_mean(x_inter, axis=1)
        y_avg = tf.reduce_mean(y_inter, axis=1)
        x_max = tf.reduce_max(x_inter, axis=1)
        y_max = tf.reduce_max(y_inter, axis=1)

        input2fc = tf.concat([x_avg, x_max, y_avg, y_max], axis=1)
        # input2fc = tf.concat([x_inter, y_inter], 2)  # (?, ?, 1024)
        # print(input2fc.shape.as_list())
        logits = self.fc(input2fc, match_dim=input2fc.shape.as_list()[-1])
        # print(logits.shape)

        # gold_matrix = tf.one_hot(labels, self.hp.num_class, dtype=tf.float32)
        # aaa = tf.print(logits, ["LOGITS", logits])
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.truth))
        with tf.variable_scope('acc', reuse=tf.AUTO_REUSE):
            label_pred = tf.argmax(logits, 1, name='label_pred')
            label_true = tf.argmax(self.truth, 1, name='label_true')
            correct_pred = tf.equal(tf.cast(label_pred, tf.int32), tf.cast(label_true, tf.int32))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='Accuracy')

        # train scheme
        global_step = tf.train.get_or_create_global_step()
        lr = noam_scheme(self.hp.lr, global_step, self.hp.warmup_steps)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

        return loss, train_op, global_step, accuracy, label_pred

    def predict_model(self):

        # representation
        x_repre, y_repre = self.representation(self.x, self.y, False)
        # y_repre = self.representation(self.y, False)

        # interactivate
        x_inter = self.interactivate(x_repre, y_repre, False)  # (?, ?, 512)
        y_inter = self.interactivate(y_repre, x_repre, False)  # (?, ?, 512)
        # print(y_inter.shape)
        # print(x_inter.shape)

        x_avg = tf.reduce_mean(x_inter, axis=1)
        y_avg = tf.reduce_mean(y_inter, axis=1)
        x_max = tf.reduce_max(x_inter, axis=1)
        y_max = tf.reduce_max(y_inter, axis=1)

        input2fc = tf.concat([x_avg, x_max, y_avg, y_max], axis=1)

        logits = self.fc(input2fc, match_dim=input2fc.shape.as_list()[-1], reuse=tf.AUTO_REUSE)

        with tf.variable_scope('acc', reuse=tf.AUTO_REUSE):
            label_pred = tf.argmax(logits, 1, name='label_pred')

        return label_pred
