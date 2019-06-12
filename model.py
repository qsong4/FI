'''
'''
import tensorflow as tf

from data_load import load_vocab
from modules import get_token_embeddings, ff, positional_encoding, multihead_attention, label_smoothing, noam_scheme
from utils import convert_idx_to_token_tensor
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)

class Transformer:
    def __init__(self, hp):
        self.hp = hp
        self.token2idx, self.idx2token = load_vocab(hp.vocab)
        self.embeddings = get_token_embeddings(self.hp.vocab_size, self.hp.d_model, zero_pad=True)

    def representation(self, xs, ys, training=True):
        with tf.variable_scope("representation", reuse=tf.AUTO_REUSE):
            x, x_seqlen = xs
            y, y_seqlen = ys

            # embedding
            encx = tf.nn.embedding_lookup(self.embeddings, x) # (N, T1, d_model)
            encx *= self.hp.d_model**0.5 # scale

            encx += positional_encoding(encx, self.hp.maxlen)
            encx = tf.layers.dropout(encx, self.hp.dropout_rate, training=training)

            ency = tf.nn.embedding_lookup(self.embeddings, y) # (N, T1, d_model)
            ency *= self.hp.d_model**0.5 # scale

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
        w = tf.get_variable("w", [match_dim, self.hp.num_class], dtype=tf.float32)
        b = tf.get_variable("b", [self.hp.num_class], dtype=tf.float32)
        logits = tf.matmul(inpt, w) + b
        #prob = tf.nn.softmax(logits)

        #gold_matrix = tf.one_hot(labels, self.hp.num_class, dtype=tf.float32)
        #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
        return logits

    def train(self, xs, ys, labels):
        # representation
        x_repre, y_repre = self.representation(xs, ys)
        #y_repre = self.representation(ys)

        # interactivate
        x_inter = self.interactivate(x_repre, y_repre)#(?, ?, 512)
        y_inter = self.interactivate(y_repre, x_repre)#(?, ?, 512)
        #print(y_inter.shape)
        #print(x_inter.shape)
        input2fc = tf.concat([x_inter, y_inter], 2)#(?, ?, 1024)
        #print(input2fc.shape)
        logits = self.fc(input2fc, match_dim=len(input2fc))

        gold_matrix = tf.one_hot(labels, self.hp.num_class, dtype=tf.float32)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=gold_matrix))

        # train scheme
        global_step = tf.train.get_or_create_global_step()
        lr = noam_scheme(self.hp.lr, global_step, self.hp.warmup_steps)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.scalar('lr', lr)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("global_step", global_step)

        summaries = tf.summary.merge_all()

        return loss, train_op, global_step, summaries

    def eval(self, xs, ys):
        '''Predicts autoregressively
        At inference, input ys is ignored.
        Returns
        y_hat: (N, T2)
        '''
        decoder_inputs, y, y_seqlen, sents2 = ys

        decoder_inputs = tf.ones((tf.shape(xs[0])[0], 1), tf.int32) * self.token2idx["<s>"]
        ys = (decoder_inputs, y, y_seqlen, sents2)

        memory, sents1 = self.encode(xs, False)

        logging.info("Inference graph is being built. Please be patient.")
        for _ in tqdm(range(self.hp.maxlen2)):
            logits, y_hat, y, sents2 = self.decode(ys, memory, False)
            if tf.reduce_sum(y_hat, 1) == self.token2idx["<pad>"]: break

            _decoder_inputs = tf.concat((decoder_inputs, y_hat), 1)
            ys = (_decoder_inputs, y, y_seqlen, sents2)

        # monitor a random sample
        n = tf.random_uniform((), 0, tf.shape(y_hat)[0]-1, tf.int32)
        sent1 = sents1[n]
        pred = convert_idx_to_token_tensor(y_hat[n], self.idx2token)
        sent2 = sents2[n]

        tf.summary.text("sent1", sent1)
        tf.summary.text("pred", pred)
        tf.summary.text("sent2", sent2)
        summaries = tf.summary.merge_all()

        return y_hat, summaries

