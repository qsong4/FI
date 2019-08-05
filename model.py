import tensorflow as tf

from data_load import load_vocab, loadGloVe, load_char_vocab
from modules import get_token_embeddings, ff, positional_encoding, multihead_attention, ln, positional_encoding_bert
from matching import match_passage_with_question, localInference
from tensorflow.python.ops import nn_ops


class FI:
    """
    xs: tuple of
        x: int32 tensor. (句子长度，)
        x_seqlens. int32 tensor. (句子)
    """

    def __init__(self, hp):
        self.hp = hp
        self.token2idx, self.idx2token, self.hp.vocab_size = load_vocab(hp.vocab)
        #_, _, self.char_vocab_size = load_char_vocab(hp.char_vocab)
        self.embd = None
        if self.hp.preembedding:
            self.embd = loadGloVe(self.hp.vec_path)
        self.embeddings = get_token_embeddings(self.embd, self.hp.vocab_size, self.hp.d_model, zero_pad=False)
        self.x = tf.placeholder(tf.int32, [None, self.hp.maxlen], name="text_x")
        self.y = tf.placeholder(tf.int32, [None, self.hp.maxlen], name="text_y")
        self.x_len = tf.placeholder(tf.int32, [None])
        self.y_len = tf.placeholder(tf.int32, [None])
        self.truth = tf.placeholder(tf.int32, [None, self.hp.num_class], name="truth")
        self.is_training = tf.placeholder(tf.bool,shape=None, name="is_training")

        if self.hp.char_embedding:
            ## Char embedding
            ## shape = (batch_size, max length of sentence, max length of word)
            self.x_char_ids = tf.placeholder(tf.int32, shape=[None, self.hp.maxlen, self.hp.char_maxlen],
                            name="x_char_ids")

            ## shape = (batch_size, max_length of sentence)
            self.x_word_lengths = tf.placeholder(tf.int32, shape=[None, self.hp.maxlen],
                            name="x_word_lengths")
            ## shape = (batch_size, max length of sentence, max length of word)
            self.y_char_ids = tf.placeholder(tf.int32, shape=[None, self.hp.maxlen, self.hp.char_maxlen],
                            name="y_char_ids")

            ## shape = (batch_size, max_length of sentence)
            self.y_word_lengths = tf.placeholder(tf.int32, shape=[None, self.hp.maxlen],
                            name="y_word_lengths")

        self.logits = self._logits_op()
        self.loss = self._loss_op()
        self.acc = self._acc_op()
        self.global_step = self._globalStep_op()
        self.train = self._training_op()

    def create_feed_dict(self, x, y, x_len, y_len, truth, is_training):
        feed_dict = {
            self.x: x,
            self.y: y,
            self.x_len: x_len,
            self.y_len: y_len,
            self.truth: truth,
            self.is_training: is_training,
        }

        return feed_dict

    def create_char_feed_dict(self, feed_dict, x_char_ids, x_word_lengths, y_char_ids, y_word_lengths):
        feed_dict[self.x_char_ids] = x_char_ids
        feed_dict[self.y_char_ids] = y_char_ids
        feed_dict[self.x_word_lengths] = x_word_lengths
        feed_dict[self.y_word_lengths] = y_word_lengths

        return feed_dict

    def create_feed_dict_infer(self, x, y, x_len, y_len):
        feed_dict = {
            self.x: x,
            self.y: y,
            self.x_len: x_len,
            self.y_len: y_len,
        }

        return feed_dict

    def my_lstm_layer(self, input_reps, lstm_dim, input_lengths=None, scope_name=None, reuse=False, is_training=True,
                      dropout_rate=0.2):
        with tf.variable_scope(scope_name, reuse=reuse):
            context_lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(lstm_dim)
            context_lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(lstm_dim)
            '''
            if is_training is not None:
                context_lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(context_lstm_cell_fw,
                                                                     output_keep_prob=(1 - dropout_rate))
                context_lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(context_lstm_cell_bw,
                                                                     output_keep_prob=(1 - dropout_rate))
            '''
            #context_lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([context_lstm_cell_fw])
            #context_lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([context_lstm_cell_bw])

            (f_rep, b_rep), _ = tf.nn.bidirectional_dynamic_rnn(
                context_lstm_cell_fw, context_lstm_cell_bw, input_reps, dtype=tf.float32,
                sequence_length=input_lengths)  # [batch_size, question_len, context_lstm_dim]
            outputs = tf.concat(axis=2, values=[f_rep, b_rep])
        return (f_rep, b_rep, outputs)

    def collect_final_step_of_lstm(self, lstm_representation, lengths):
        # lstm_representation: [batch_size, passsage_length, dim]
        # lengths: [batch_size]
        lengths = tf.maximum(lengths, tf.zeros_like(lengths, dtype=tf.int32))

        batch_size = tf.shape(lengths)[0]
        batch_nums = tf.range(0, limit=batch_size)  # shape (batch_size)
        indices = tf.stack((batch_nums, lengths), axis=1)  # shape (batch_size, 2)
        result = tf.gather_nd(lstm_representation, indices, name='last-forwar-lstm')
        return result  # [batch_size, dim]

    def _char_embedding(self):
        with tf.variable_scope("chars", reuse=tf.AUTO_REUSE):
            # get embeddings matrix
            _char_embeddings = tf.get_variable(name="_char_embeddings", dtype=tf.float32,
                shape=[self.char_vocab_size, self.hp.char_dim])

            x_char_embeddings = tf.nn.embedding_lookup(_char_embeddings, self.x_char_ids,
                name="x_char_embeddings")
            y_char_embeddings = tf.nn.embedding_lookup(_char_embeddings, self.y_char_ids,
                                                       name="y_char_embeddings")
            # put the time dimension on axis=1
            s_x = tf.shape(x_char_embeddings)
            s_y = tf.shape(y_char_embeddings)
            x_char_embeddings = tf.reshape(x_char_embeddings, shape=[-1, s_x[-2], self.hp.char_dim])
            y_char_embeddings = tf.reshape(y_char_embeddings, shape=[-1, s_y[-2], self.hp.char_dim])
            x_lengths = tf.reshape(self.x_word_lengths, shape=[-1])
            y_lengths = tf.reshape(self.y_word_lengths, shape=[-1])
            x_char_mask = tf.sequence_mask(x_lengths, s_x[2], dtype=tf.float32)  # [batch_size*question_len, q_char_len]
            x_char_embeddings = tf.multiply(x_char_embeddings, tf.expand_dims(x_char_mask, axis=-1))
            y_char_mask = tf.sequence_mask(y_lengths, s_y[2], dtype=tf.float32)  # [batch_size*question_len, q_char_len]
            y_char_embeddings = tf.multiply(y_char_embeddings, tf.expand_dims(y_char_mask, axis=-1))

            # bi lstm on chars
            # need 2 instances of cells since tf 1.1
            (x_char_fw, x_char_bw, _) = self.my_lstm_layer(x_char_embeddings, self.hp.char_lstm_dim, input_lengths=x_lengths,
                                                    scope_name="char_lstm", reuse=tf.AUTO_REUSE, is_training=self.is_training,
                                                    dropout_rate=self.hp.dropout_rate)
            (y_char_fw, y_char_bw, _) = self.my_lstm_layer(y_char_embeddings, self.hp.char_lstm_dim, input_lengths=y_lengths,
                                                    scope_name="char_lstm", reuse=tf.AUTO_REUSE, is_training=self.is_training,
                                                    dropout_rate=self.hp.dropout_rate)

            x_char_fw = self.collect_final_step_of_lstm(x_char_fw, x_lengths-1)
            x_char_bw = x_char_bw[:, 0, :]
            y_char_fw = self.collect_final_step_of_lstm(y_char_fw, y_lengths-1)
            y_char_bw = y_char_bw[:, 0, :]

            x_char_output = tf.concat(axis=1, values=[x_char_fw, x_char_bw])
            y_char_output = tf.concat(axis=1, values=[y_char_fw, y_char_bw])

            # shape = (batch size, max sentence length, char hidden size)
            self.ppp = tf.print(["x_char_output"], x_char_output.shape)
            x_char_output = tf.reshape(x_char_output, shape=[self.hp.batch_size, s_x[1], 2 * self.hp.char_lstm_dim])

            y_char_output = tf.reshape(y_char_output, shape=[self.hp.batch_size, s_y[1], 2 * self.hp.char_lstm_dim])
            #print(x_char_output.shape)
            #print(x_char_output.shape)
            return x_char_output, y_char_output

    def representation(self, xs, ys):
        with tf.variable_scope("representation", reuse=tf.AUTO_REUSE):
            x = xs
            y = ys

            # print(x)
            # print(y)

            # embedding
            #x_char_emb, y_char_emb = self._char_embedding()

            encx = tf.nn.embedding_lookup(self.embeddings, x)  # (N, T1, d_model)
            encx *= self.hp.d_model ** 0.5  # scale

            #encx += positional_encoding(encx, self.hp.maxlen)
            encx += positional_encoding_bert(encx, self.hp.maxlen)#bert版本的position embedding
            #encx = tf.concat([encx, x_char_emb], axis=-1)
            encx = tf.layers.dropout(encx, self.hp.dropout_rate, training=self.is_training)

            ency = tf.nn.embedding_lookup(self.embeddings, y)  # (N, T1, d_model)
            ency *= self.hp.d_model ** 0.5  # scale

            #ency += positional_encoding(ency, self.hp.maxlen)
            ency += positional_encoding_bert(ency, self.hp.maxlen)
            #ency = tf.concat([ency, y_char_emb], axis=-1)
            ency = tf.layers.dropout(ency, self.hp.dropout_rate, training=self.is_training)

            # add ln
            encx = ln(encx)
            ency = ln(ency)

            # 这两个模块可以互换
            # Inter Inference Block
            for i in range(self.hp.num_inter_blocks):
                encx, ency = self.inter_blocks(encx, ency, scope="num_inter_blocks_{}".format(i))

            #print(encx.shape)
            #print(ency.shape)
            inter_encx = encx
            inter_ency = ency

            # Inference Block
            for i in range(self.hp.inference_blocks):
                encx, ency = self.inference_blocks(encx, ency, scope="num_inference_blocks_{}".format(i))

            #将inter的结果和infer的结果拼接
            encx = tf.concat([encx, inter_encx], axis=-1)
            ency = tf.concat([ency, inter_ency], axis=-1)

            encx = tf.layers.dropout(encx, self.hp.dropout_rate, training=self.is_training)
            ency = tf.layers.dropout(ency, self.hp.dropout_rate, training=self.is_training)

        # return x_layer, y_layer
        return encx, ency

    # def match_sentences(self, x_steps, t_stepss):

    def inference_blocks(self, a_repre, b_repre, scope, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(scope, reuse=reuse):
            _encx = multihead_attention(queries=a_repre,
                                       keys=a_repre,
                                       values=a_repre,
                                       num_heads=self.hp.num_heads,
                                       dropout_rate=self.hp.dropout_rate,
                                       training=self.is_training,
                                       causality=False)

            # self-attention
            _ency = multihead_attention(queries=b_repre,
                                       keys=b_repre,
                                       values=b_repre,
                                       num_heads=self.hp.num_heads,
                                       dropout_rate=self.hp.dropout_rate,
                                       training=self.is_training,
                                       causality=False)

            # self-attention

            encx = multihead_attention(queries=_ency,
                                       keys=_encx,
                                       values=_encx,
                                       num_heads=self.hp.num_heads,
                                       dropout_rate=self.hp.dropout_rate,
                                       training=self.is_training,
                                       causality=False)

            # self-attention
            ency = multihead_attention(queries=_encx,
                                       keys=_ency,
                                       values=_ency,
                                       num_heads=self.hp.num_heads,
                                       dropout_rate=self.hp.dropout_rate,
                                       training=self.is_training,
                                       causality=False)

            encx, ency = self._infer(encx, ency)
            dim = encx.shape.as_list()[-1]
            # feed forward
            encx = ff(encx, num_units=[self.hp.d_ff, dim])
            ency = ff(ency, num_units=[self.hp.d_ff, dim])

            #先进行infer然后再过全连接
            #encx = ff(encx, num_units=[self.hp.d_ff, self.hp.d_model])
            #ency = ff(ency, num_units=[self.hp.d_ff, self.hp.d_model])

            return encx, ency

    def _infer(self, encx, ency, scope="local_inference"):
        with tf.variable_scope(scope):
            #x_mask = tf.sequence_mask(self.x_len, self.hp.maxlen, dtype=tf.float32)
            #y_mask = tf.sequence_mask(self.y_len, self.hp.maxlen, dtype=tf.float32)
            # match_result_x = match_passage_with_question(encx, ency, x_mask, y_mask)
            # match_result_y = match_passage_with_question(ency, encx, x_mask, y_mask)

            # attentionWeights = self.calcuate_attention(encx, ency, self.hp.d_model+self.hp.char_lstm_dim*2, self.hp.d_model+self.hp.char_lstm_dim*2,
            #                                   scope_name="attention", att_type=self.hp.att_type, att_dim=self.hp.att_dim,
            #                                   remove_diagnoal=False, mask1=x_mask, mask2=y_mask)

            attentionWeights = tf.matmul(encx, tf.transpose(ency, [0, 2, 1]))
            attentionSoft_a = tf.nn.softmax(attentionWeights)
            attentionSoft_b = tf.nn.softmax(tf.transpose(attentionWeights))
            attentionSoft_b = tf.transpose(attentionSoft_b)

            a_hat = tf.matmul(attentionSoft_a, ency)
            b_hat = tf.matmul(attentionSoft_b, encx)

            a_diff = tf.subtract(encx, a_hat)
            a_mul = tf.multiply(encx, a_hat)
            b_diff = tf.subtract(ency, b_hat)
            b_mul = tf.multiply(ency, b_hat)

            a_res = tf.concat([a_hat, a_diff, a_mul], axis=2)
            b_res = tf.concat([b_hat, b_diff, b_mul], axis=2)

            # BN
            # a_res = tf.layers.batch_normalization(a_res, training=self.is_training, name='bn1', reuse=tf.AUTO_REUSE)
            # b_res = tf.layers.batch_normalization(b_res, training=self.is_training, name='bn2', reuse=tf.AUTO_REUSE)
            # project
            a_res = self._project_op(a_res)  # (?,?,d_model)
            b_res = self._project_op(b_res)  # (?,?,d_model)

            a_res += encx
            b_res += ency

            a_res = ln(a_res)
            b_res = ln(b_res)

        return a_res, b_res

    def _project_op(self, inputx):
        #dim = inputx.shape.as_list()[-1]
        with tf.variable_scope("projection", reuse=tf.AUTO_REUSE):
            inputx = tf.layers.dense(inputx, self.hp.d_model,
                                     activation=tf.nn.relu,
                                     name='fnn',
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

            return inputx

    def base_blocks(self, a_repre, scope, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(scope, reuse=reuse):
            # self-attention
            encx = multihead_attention(queries=a_repre,
                                       keys=a_repre,
                                       values=a_repre,
                                       num_heads=self.hp.num_heads,
                                       dropout_rate=self.hp.dropout_rate,
                                       training=self.is_training,
                                       causality=False)
            # feed forward
            encx = ff(encx, num_units=[self.hp.d_ff, self.hp.d_model])
        return encx

    def inter_blocks(self, a_repre, b_repre, scope, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(scope, reuse=reuse):
            # self-attention
            dim = a_repre.shape.as_list()[-1]
            encx = multihead_attention(queries=a_repre,
                                       keys=a_repre,
                                       values=a_repre,
                                       num_heads=self.hp.num_heads,
                                       dropout_rate=self.hp.dropout_rate,
                                       training=self.is_training,
                                       causality=False)
            # feed forward
            encx = ff(encx, num_units=[self.hp.d_ff, dim])

            # self-attention
            ency = multihead_attention(queries=b_repre,
                                       keys=b_repre,
                                       values=b_repre,
                                       num_heads=self.hp.num_heads,
                                       dropout_rate=self.hp.dropout_rate,
                                       training=self.is_training,
                                       causality=False)
            # feed forward
            ency = ff(ency, num_units=[self.hp.d_ff, dim])
            #print(encx.shape)
            #print(ency.shape)
            #encx, ency = self._infer(encx, ency)

        return encx, ency

    def aggregation(self, a_repre, b_repre):
        dim = a_repre.shape.as_list()[-1]
        with tf.variable_scope("aggregation", reuse=tf.AUTO_REUSE):
            # Blocks
            for i in range(self.hp.num_agg_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # Vanilla attention
                    a_repre = multihead_attention(queries=a_repre,
                                                  keys=a_repre,
                                                  values=a_repre,
                                                  num_heads=self.hp.num_heads,
                                                  dropout_rate=self.hp.dropout_rate,
                                                  training=self.is_training,
                                                  causality=False,
                                                  scope="vanilla_attention")
                    ### Feed Forward
                    a_repre = ff(a_repre, num_units=[self.hp.d_ff, dim])
        return a_repre

    def fc(self, inpt, match_dim, reuse=tf.AUTO_REUSE):
        with tf.variable_scope("fc", reuse=reuse):
            w = tf.get_variable("w", [match_dim, self.hp.num_class], dtype=tf.float32)
            b = tf.get_variable("b", [self.hp.num_class], dtype=tf.float32)
            logits = tf.matmul(inpt, w) + b
        # prob = tf.nn.softmax(logits)

        # gold_matrix = tf.one_hot(labels, self.hp.num_class, dtype=tf.float32)
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
        return logits

    def fc_2l(self, inputs, num_units, scope="fc_2l"):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # Inner layer
            outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)

            # Outer layer
            outputs = tf.layers.dense(outputs, num_units[1])

        return outputs
    # calculate classification accuracy
    def _acc_op(self):
        with tf.name_scope('acc'):
            label_pred = tf.argmax(self.logits, 1, name='label_pred')
            label_true = tf.argmax(self.y, 1, name='label_true')
            correct_pred = tf.equal(tf.cast(label_pred, tf.int32), tf.cast(label_true, tf.int32))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='Accuracy')
        return accuracy

    # 将transformer的每层作为一个channel输入CNN
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

    def _logits_op(self):
        # representation
        x_repre, y_repre = self.representation(self.x, self.y)  # (layers, batchsize, maxlen, d_model)

        # BN
        x_repre = ln(x_repre)
        y_repre = ln(y_repre)
        # x_repre = tf.layers.batch_normalization(x_repre, training=self.is_training, name='bn1', reuse=tf.AUTO_REUSE)
        # y_repre = tf.layers.batch_normalization(y_repre, training=self.is_training, name='bn2', reuse=tf.AUTO_REUSE)

        # aggre
        # x_repre = self.aggregation(x_repre, x_repre)
        # y_repre = self.aggregation(y_repre, y_repre)

        avg_x = tf.reduce_mean(x_repre, axis=1)
        # max_x = tf.reduce_max(x_repre, axis=1)
        avg_y = tf.reduce_mean(y_repre, axis=1)
        # max_y = tf.reduce_max(y_repre, axis=1)
        #agg_res = tf.concat([avg_x, avg_y, max_x, max_y], axis=1)
        agg_res = tf.concat([avg_x, avg_y], axis=1)
        #logits = self.fc(agg_res, match_dim=agg_res.shape.as_list()[-1])
        logits = self.fc_2l(agg_res, num_units=[self.hp.d_ff, self.hp.num_class])
        return logits

    def _loss_op(self, l2_lambda=0.0001, label_smoothing=0.1):
        self.truth = tf.cast(self.truth, self.logits.dtype)
        if label_smoothing > 0:
            smooth_positives = 1.0 - label_smoothing
            smooth_negatives = label_smoothing / self.hp.num_class
            self.truth = self.truth * smooth_positives + smooth_negatives
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.truth))
        weights = [v for v in tf.trainable_variables() if ('w' in v.name) or ('kernel') in v.name]
        l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights]) * l2_lambda
        loss += l2_loss
        return loss

    def _acc_op(self):
        with tf.variable_scope('acc', reuse=tf.AUTO_REUSE):
            label_pred = tf.argmax(self.logits, 1, name='label_pred')
            label_true = tf.argmax(self.truth, 1, name='label_true')
            correct_pred = tf.equal(tf.cast(label_pred, tf.int32), tf.cast(label_true, tf.int32))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='Accuracy')
        return accuracy

    def _globalStep_op(self):
        global_step = tf.train.get_or_create_global_step()
        return global_step

    def _training_op(self):
        # train scheme
        # global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer(self.hp.lr)
        # optimizer = tf.train.AdadeltaOptimizer(lr)

        '''
        if self.hp.lambda_l2>0.0:
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
            loss = loss + self.hp.lambda_l2 * l2_loss
        '''

        # grads = self.compute_gradients(loss, tvars)
        # grads, _ = tf.clip_by_global_norm(grads, 10.0)
        # train_op = optimizer.minimize(loss, global_step=global_step)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # train_op = optimizer.minimize(loss)
            train_op = optimizer.minimize(self.loss, global_step=self.global_step)

        return train_op

    def predict_model(self):
        # representation
        x_repre, y_repre = self.representation(self.x, self.y)  # (batchsize, maxlen, d_model)
        x_mask = tf.sequence_mask(self.x_len, self.hp.maxlen, dtype=tf.float32)
        y_mask = tf.sequence_mask(self.y_len, self.hp.maxlen, dtype=tf.float32)

        # matching
        match_result = match_passage_with_question(x_repre, y_repre, x_mask, y_mask)

        # aggre
        x_inter = self.aggregation(match_result, match_result)  # (?, ?, 512)

        x_avg = tf.reduce_mean(x_inter, axis=1)
        x_max = tf.reduce_max(x_inter, axis=1)

        input2fc = tf.concat([x_avg, x_max], axis=1)
        logits = self.fc(input2fc, match_dim=input2fc.shape.as_list()[-1])

        with tf.variable_scope('acc', reuse=tf.AUTO_REUSE):
            label_pred = tf.argmax(logits, 1, name='label_pred')

        return label_pred
