import tensorflow as tf

from data_load import load_vocab, loadGloVe
from modules import get_token_embeddings, ff, positional_encoding, multihead_attention, ln, noam_scheme
from matching import match_passage_with_question

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

    def create_feed_dict_infer(self, x, y, x_len, y_len):
        feed_dict = {
            self.x: x,
            self.y: y,
            self.x_len: x_len,
            self.y_len: y_len,
        }

        return feed_dict


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

            #add ln
            encx = ln(encx)
            ency = ln(ency)

            ## Blocks
            x_layer = []
            y_layer = []
            for i in range(self.hp.num_extract_blocks + self.hp.num_inter_blocks):
                if i < self.hp.num_extract_blocks:
                    encx = self.base_blocks(encx, encx, training=training, scope="num_blocks_{}".format(i))
                    ency = self.base_blocks(ency, ency, training=training, scope="num_blocks_{}".format(i))
                    x_layer.append(encx)
                    y_layer.append(ency)
                else:
                    encx, ency = self.inter_blocks(encx, ency, training=training, scope="num_blocks_{}".format(i))
                    x_layer.append(encx)
                    y_layer.append(ency)
        return x_layer, y_layer
        #return encx, ency

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

    def aggregation(self, a_repre, b_repre, training=True):
        dim = a_repre.shape.as_list()[-1]
        with tf.variable_scope("aggregation", reuse=tf.AUTO_REUSE):
            # Blocks
            for i in range(self.hp.num_agg_blocks):
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
                    dec = ff(dec, num_units=[self.hp.d_ff, dim])
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

    #将transformer的每层作为一个channel输入CNN
    def cnn_agg(self, match_channels):
        # Create a convolution + maxpool layer for each filter size
        filter_sizes = list(map(int, self.hp.filter_sizes.split(",")))
        embedding_size = match_channels.shape.as_list()[2]
        sequence_length = match_channels.shape.as_list()[1]
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 6, self.hp.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.hp.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    match_channels,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.hp.num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat, self.hp.dropout_rate)

        return h_drop

    def build_model(self):
        # representation
        x_repre_list, y_repre_list = self.representation(self.x, self.y) #(layers, batchsize, maxlen, d_model)
        x_mask = tf.sequence_mask(self.x_len, self.hp.maxlen, dtype=tf.float32)
        y_mask = tf.sequence_mask(self.y_len, self.hp.maxlen, dtype=tf.float32)

        match_result = []
        for x_repre, y_repre in zip(x_repre_list, y_repre_list):
            match_result.append(match_passage_with_question(x_repre, y_repre, x_mask, y_mask))

        match_result = tf.concat(axis=2, values=match_result)
        #match_stack = tf.stack(match_result, -1) # (batchsize, maxlen, mp_dim, channels)
        # aggre
        #agg_res = self.cnn_agg(match_stack)
        agg_res = self.aggregation(match_result, match_result)
        avg_res = tf.reduce_mean(agg_res, axis=1)
        max_res = tf.reduce_max(agg_res, axis=1)
        agg_res = tf.concat([avg_res, max_res], axis=1)
        logits = self.fc(agg_res, match_dim=agg_res.shape.as_list()[-1])

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
        #optimizer = tf.train.AdadeltaOptimizer(lr)

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

    def eval_model(self, is_training=True):
        # representation
        x_repre_list, y_repre_list = self.representation(self.x, self.y) #(layers, batchsize, maxlen, d_model)
        x_mask = tf.sequence_mask(self.x_len, self.hp.maxlen, dtype=tf.float32)
        y_mask = tf.sequence_mask(self.y_len, self.hp.maxlen, dtype=tf.float32)

        match_result = []
        for x_repre, y_repre in zip(x_repre_list, y_repre_list):
            match_result.append(match_passage_with_question(x_repre, y_repre, x_mask, y_mask))
        match_stack = tf.stack(match_result, -1) # (batchsize, maxlen, mp_dim, channels)
        # aggre
        agg_res = self.cnn_agg(match_stack)

        logits = self.fc(agg_res, match_dim=agg_res.shape.as_list()[-1])

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.truth))
        with tf.variable_scope('acc', reuse=tf.AUTO_REUSE):
            label_pred = tf.argmax(logits, 1, name='label_pred')
            label_true = tf.argmax(self.truth, 1, name='label_true')
            correct_pred = tf.equal(tf.cast(label_pred, tf.int32), tf.cast(label_true, tf.int32))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='Accuracy')


        return loss,  accuracy

    def predict_model(self):
        # representation
        x_repre, y_repre = self.representation(self.x, self.y, False) # (batchsize, maxlen, d_model)
        x_mask = tf.sequence_mask(self.x_len, self.hp.maxlen, dtype=tf.float32)
        y_mask = tf.sequence_mask(self.y_len, self.hp.maxlen, dtype=tf.float32)


        # matching
        match_result = self.match_passage_with_question(x_repre, y_repre, x_mask, y_mask)

        # aggre
        x_inter = self.aggregation(match_result, match_result, False)  # (?, ?, 512)

        x_avg = tf.reduce_mean(x_inter, axis=1)
        x_max = tf.reduce_max(x_inter, axis=1)

        input2fc = tf.concat([x_avg, x_max], axis=1)
        logits = self.fc(input2fc, match_dim=input2fc.shape.as_list()[-1])

        with tf.variable_scope('acc', reuse=tf.AUTO_REUSE):
            label_pred = tf.argmax(logits, 1, name='label_pred')

        return label_pred





