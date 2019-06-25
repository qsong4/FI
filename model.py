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
        self.x_len = tf.placeholder(tf.int32, [None])
        self.y_len = tf.placeholder(tf.int32, [None])
        self.truth = tf.placeholder(tf.int32, [None, self.hp.num_class], name="truth")

    def create_feed_dict(self, x, y, x_len, y_len, truth):
        feed_dict = {
            self.x: x,
            self.y: y,
            self.x_len: x_len,
            self.y_len: y_len,
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

    def cosine_distance(self, x1, x2, cosine_norm=True, eps=1e-6):
        cosine_numerator = tf.reduce_sum(tf.multiply(x1, x2), axis=-1)
        if not cosine_norm:
            return tf.tanh(cosine_numerator)
        x1_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(x1), axis=-1), eps))
        x2_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(x2), axis=-1), eps))
        return cosine_numerator / x1_norm / x2_norm

    def cal_relevancy_matrix(self, x, y):
        x_temp = tf.expand_dims(x, 1)
        y_temp = tf.expand_dims(y, 2)
        relevancy_matrix = self.cosine_distance(x_temp, y_temp, cosine_norm=True)
        return relevancy_matrix

    def mask_relevancy_matrix(self, relevancy_matrix, x_mask, y_mask):
        relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(x_mask, axis=1))
        relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(y_mask, axis=2))
        return relevancy_matrix

    def match_passage_with_question(self, x1, x2, x1_mask, x2_mask, scope="match_x_with_y"):
        x_repre = tf.multiply(x1, tf.expand_dims(x1_mask, axis=-1))
        y_repre = tf.multiply(x2, tf.expand_dims(x2_mask, axis=-1))
        all_x_aware_y_representation = []
        with tf.variable_scope(scope or "match_x_with_y"):
            relevancy_matrix = self.cal_relevancy_matrix(x_repre, y_repre)
            relevancy_matrix = self.mask_relevancy_matrix(relevancy_matrix, x1_mask, x2_mask)
            all_x_aware_y_representation.append(tf.reduce_max(relevancy_matrix, axis=2, keepdims=True))
            all_x_aware_y_representation.append(tf.reduce_mean(relevancy_matrix, axis=2, keepdims=True))

            attentive_rep = self.multi_perspective_matching(x_repre, y_repre)
            all_x_aware_y_representation.append(attentive_rep)

        all_x_aware_y_representation = tf.concat(axis=2, values=all_x_aware_y_representation)

        return all_x_aware_y_representation

    def multi_perspective_matching(self, repre1, repre2, scope='mp_match', reuse=tf.AUTO_REUSE):
        input_shape = tf.shape(repre1)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        matching_result = []

        with tf.variable_scope(scope, reuse=reuse):
            if self.hp.with_cosine:
                cosine_value = self.cosine_distance(repre1, repre2, cosine_norm=False)
                cosine_value = tf.reshape(cosine_value, [batch_size, seq_length, 1])
                matching_result.append(cosine_value)
            if self.hp.with_mp_cosine:
                mp_cosine_params = tf.get_variable("mp_cosine", shape=[self.hp.cosine_MP_dim, self.hp.d_model], dtype=tf.float32)
                mp_cosine_params = tf.expand_dims(mp_cosine_params, axis=0)
                mp_cosine_params = tf.expand_dims(mp_cosine_params, axis=0)

                repre1_flat = tf.expand_dims(repre1, axis=2)
                repre2_flat = tf.expand_dims(repre2, axis=2)
                #print(repre1_flat.shape)
                #print(mp_cosine_params.shape)
                mp_cosine_matching = self.cosine_distance(tf.multiply(repre1_flat, mp_cosine_params),repre2_flat)
                matching_result.append(mp_cosine_matching)
        matching_result = tf.concat(axis=2, values=matching_result)
        return matching_result

    def representation(self, xs, ys, training=True):
        #x_steps = []
        #y_steps = []
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

            ## Blocks
            for i in range(self.hp.num_blocks):
                if i == 0:
                    encx = self.base_blocks(encx, encx, scope="num_blocks_{}".format(i))
                    ency = self.base_blocks(ency, ency, scope="num_blocks_{}".format(i))
                else:
                    encx, ency = self.inter_blocks(encx, ency, scope="num_blocks_{}".format(i))
                    #ency = self.inter_blocks(ency, encx, scope="num_blocks_{}".format(i))
        return encx, ency

    #def match_sentences(self, x_steps, t_stepss):


    def base_blocks(self, a_repre, b_repre, scope, training=True, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(scope, reuse=reuse):
            # self-attention
            encx = multihead_attention(queries=a_repre,
                                       keys=b_repre,
                                       values=b_repre,
                                       num_heads=self.hp.num_heads,
                                       dropout_rate=self.hp.dropout_rate,
                                       training=training,
                                       causality=False)
            # feed forward
            encx = ff(encx, num_units=[self.hp.d_ff, self.hp.d_model])
        return encx

    def inter_blocks(self, a_repre, b_repre, scope, training=True, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(scope, reuse=reuse):
            # self-attention
            encx = multihead_attention(queries=b_repre,
                                       keys=a_repre,
                                       values=a_repre,
                                       num_heads=self.hp.num_heads,
                                       dropout_rate=self.hp.dropout_rate,
                                       training=training,
                                       causality=False)
            # feed forward
            encx = ff(encx, num_units=[self.hp.d_ff, self.hp.d_model])

            # self-attention
            ency = multihead_attention(queries=a_repre,
                                       keys=b_repre,
                                       values=b_repre,
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
                    dec = ff(dec, num_units=[self.hp.d_ff, self.hp.cosine_MP_dim+3])
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
        x_repre, y_repre = self.representation(self.x, self.y) # (batchsize, maxlen, d_model)
        x_mask = tf.sequence_mask(self.x_len, self.hp.maxlen, dtype=tf.float32)
        y_mask = tf.sequence_mask(self.y_len, self.hp.maxlen, dtype=tf.float32)


        # matching
        match_result = self.match_passage_with_question(x_repre, y_repre, x_mask, y_mask)

        # aggre
        x_inter = self.interactivate(match_result, match_result)  # (?, ?, 512)

        x_avg = tf.reduce_mean(x_inter, axis=1)
        x_max = tf.reduce_max(x_inter, axis=1)

        input2fc = tf.concat([x_avg, x_max], axis=1)
        logits = self.fc(input2fc, match_dim=input2fc.shape.as_list()[-1])

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

        tvars = tf.trainable_variables()
        '''
        if self.hp.lambda_l2>0.0:
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
            loss = loss + self.hp.lambda_l2 * l2_loss
        '''

        # grads = self.compute_gradients(loss, tvars)
        # grads, _ = tf.clip_by_global_norm(grads, 10.0)
        # train_op = optimizer.minimize(loss, global_step=global_step)

        train_op = optimizer.minimize(loss, global_step=global_step)

        return loss, train_op, global_step, accuracy, label_pred

    def eval_model(self):
        # representation
        x_repre, y_repre = self.representation(self.x, self.y) # (batchsize, maxlen, d_model)
        x_mask = tf.sequence_mask(self.x_len, self.hp.maxlen, dtype=tf.float32)
        y_mask = tf.sequence_mask(self.y_len, self.hp.maxlen, dtype=tf.float32)


        # matching
        match_result = self.match_passage_with_question(x_repre, y_repre, x_mask, y_mask)

        # aggre
        x_inter = self.interactivate(match_result, match_result)  # (?, ?, 512)

        x_avg = tf.reduce_mean(x_inter, axis=1)
        x_max = tf.reduce_max(x_inter, axis=1)

        input2fc = tf.concat([x_avg, x_max], axis=1)
        logits = self.fc(input2fc, match_dim=input2fc.shape.as_list()[-1])

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.truth))
        with tf.variable_scope('acc', reuse=tf.AUTO_REUSE):
            label_pred = tf.argmax(logits, 1, name='label_pred')
            label_true = tf.argmax(self.truth, 1, name='label_true')
            correct_pred = tf.equal(tf.cast(label_pred, tf.int32), tf.cast(label_true, tf.int32))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='Accuracy')


        return loss,  accuracy

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





