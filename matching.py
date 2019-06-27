import tensorflow as tf
from tensorflow.python.ops import nn_ops
from hparams import Hparams
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()

def calcuate_attention(in_value_1, in_value_2, feature_dim1, feature_dim2, scope_name='att',
                       att_type='symmetric', att_dim=20, remove_diagnoal=False, mask1=None, mask2=None, training=False):
    input_shape = tf.shape(in_value_1)
    batch_size = input_shape[0]
    len_1 = input_shape[1]
    len_2 = tf.shape(in_value_2)[1]

    in_value_1 = tf.layers.dropout(in_value_1, hp.dropout_rate, training=training)
    in_value_2 = tf.layers.dropout(in_value_2, hp.dropout_rate, training=training)
    with tf.variable_scope(scope_name):
        # calculate attention ==> a: [batch_size, len_1, len_2]
        atten_w1 = tf.get_variable("atten_w1", [feature_dim1, att_dim], dtype=tf.float32)
        if feature_dim1 == feature_dim2: atten_w2 = atten_w1
        else: atten_w2 = tf.get_variable("atten_w2", [feature_dim2, att_dim], dtype=tf.float32)
        atten_value_1 = tf.matmul(tf.reshape(in_value_1, [batch_size * len_1, feature_dim1]), atten_w1)  # [batch_size*len_1, feature_dim]
        atten_value_1 = tf.reshape(atten_value_1, [batch_size, len_1, att_dim])
        atten_value_2 = tf.matmul(tf.reshape(in_value_2, [batch_size * len_2, feature_dim2]), atten_w2)  # [batch_size*len_2, feature_dim]
        atten_value_2 = tf.reshape(atten_value_2, [batch_size, len_2, att_dim])


        if att_type == 'additive':
            atten_b = tf.get_variable("atten_b", [att_dim], dtype=tf.float32)
            atten_v = tf.get_variable("atten_v", [1, att_dim], dtype=tf.float32)
            atten_value_1 = tf.expand_dims(atten_value_1, axis=2, name="atten_value_1")  # [batch_size, len_1, 'x', feature_dim]
            atten_value_2 = tf.expand_dims(atten_value_2, axis=1, name="atten_value_2")  # [batch_size, 'x', len_2, feature_dim]
            atten_value = atten_value_1 + atten_value_2  # + tf.expand_dims(tf.expand_dims(tf.expand_dims(atten_b, axis=0), axis=0), axis=0)
            atten_value = nn_ops.bias_add(atten_value, atten_b)
            atten_value = tf.tanh(atten_value)  # [batch_size, len_1, len_2, feature_dim]
            atten_value = tf.reshape(atten_value, [-1, att_dim]) * atten_v  # tf.expand_dims(atten_v, axis=0) # [batch_size*len_1*len_2, feature_dim]
            atten_value = tf.reduce_sum(atten_value, axis=-1)
            atten_value = tf.reshape(atten_value, [batch_size, len_1, len_2])
        else:
            atten_value_1 = tf.tanh(atten_value_1)
            # atten_value_1 = tf.nn.relu(atten_value_1)
            atten_value_2 = tf.tanh(atten_value_2)
            # atten_value_2 = tf.nn.relu(atten_value_2)
            diagnoal_params = tf.get_variable("diagnoal_params", [1, 1, att_dim], dtype=tf.float32)
            atten_value_1 = atten_value_1 * diagnoal_params
            atten_value = tf.matmul(atten_value_1, atten_value_2, transpose_b=True) # [batch_size, len_1, len_2]

        # normalize
        if remove_diagnoal:
            diagnoal = tf.ones([len_1], tf.float32)  # [len1]
            diagnoal = 1.0 - tf.diag(diagnoal)  # [len1, len1]
            diagnoal = tf.expand_dims(diagnoal, axis=0)  # ['x', len1, len1]
            atten_value = atten_value * diagnoal
        if mask1 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask1, axis=-1))
        if mask2 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask2, axis=1))
        atten_value = tf.nn.softmax(atten_value, name='atten_value')  # [batch_size, len_1, len_2]
        if remove_diagnoal: atten_value = atten_value * diagnoal
        if mask1 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask1, axis=-1))
        if mask2 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask2, axis=1))

    return atten_value

def make_attention_mat(x1, x2):
    euclidean = tf.sqrt(tf.reduce_sum(tf.square(x1 - tf.matrix_transpose(x2)), axis=1) + 1e-8)
    return 1 / (1 + euclidean)


def cosine_distance(x1, x2, cosine_norm=True, eps=1e-6):
    cosine_numerator = tf.reduce_sum(tf.multiply(x1, x2), axis=-1)
    if not cosine_norm:
        return tf.tanh(cosine_numerator)
    x1_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(x1), axis=-1), eps))
    x2_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(x2), axis=-1), eps))
    return cosine_numerator / x1_norm / x2_norm


def cal_relevancy_matrix(x, y):
    x_temp = tf.expand_dims(x, 1)
    y_temp = tf.expand_dims(y, 2)
    relevancy_matrix = cosine_distance(x_temp, y_temp, cosine_norm=True)
    return relevancy_matrix


def mask_relevancy_matrix(relevancy_matrix, x_mask, y_mask):
    relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(x_mask, axis=1))
    relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(y_mask, axis=2))
    return relevancy_matrix

def multi_perspective_expand_for_2D(in_tensor, decompose_params):
    in_tensor = tf.expand_dims(in_tensor, axis=1) #[batch_size, 'x', dim]
    decompose_params = tf.expand_dims(decompose_params, axis=0) # [1, decompse_dim, dim]
    return tf.multiply(in_tensor, decompose_params) # [batch_size, decompse_dim, dim]

def collect_representation(representation, positions):
    # representation: [batch_size, node_num, feature_dim]
    # positions: [batch_size, neigh_num]
    return collect_probs(representation, positions)

def collect_probs(probs, positions):
    # probs [batch_size, chunks_size]
    # positions [batch_size, pair_size]
    batch_size = tf.shape(probs)[0]
    pair_size = tf.shape(positions)[1]
    batch_nums = tf.range(0, limit=batch_size) # shape (batch_size)
    batch_nums = tf.reshape(batch_nums, shape=[-1, 1]) # [batch_size, 1]
    batch_nums = tf.tile(batch_nums, multiples=[1, pair_size]) # [batch_size, pair_size]

    indices = tf.stack((batch_nums, positions), axis=2) # shape (batch_size, pair_size, 2)
    pair_probs = tf.gather_nd(probs, indices)
    # pair_probs = tf.reshape(pair_probs, shape=[batch_size, pair_size])
    return pair_probs

def cal_max_question_representation(question_representation, atten_scores):
    atten_positions = tf.argmax(atten_scores, axis=2, output_type=tf.int32)  # [batch_size, passage_len]
    max_question_reps = collect_representation(question_representation, atten_positions)
    return max_question_reps

def cal_maxpooling_matching(x1, x2, decompose_params):
    # passage_representation: [batch_size, passage_len, dim]
    # qusetion_representation: [batch_size, question_len, dim]
    # decompose_params: [decompose_dim, dim]

    def singel_instance(x):
        p = x[0]
        q = x[1]
        # p: [pasasge_len, dim], q: [question_len, dim]
        p = multi_perspective_expand_for_2D(p, decompose_params)  # [pasasge_len, decompose_dim, dim]
        q = multi_perspective_expand_for_2D(q, decompose_params)  # [question_len, decompose_dim, dim]
        p = tf.expand_dims(p, 1)  # [pasasge_len, 1, decompose_dim, dim]
        q = tf.expand_dims(q, 0)  # [1, question_len, decompose_dim, dim]
        return cosine_distance(p, q)  # [passage_len, question_len, decompose]

    elems = (x1, x2)
    matching_matrix = tf.map_fn(singel_instance, elems,
                                dtype=tf.float32)  # [batch_size, passage_len, question_len, decompse_dim]
    return tf.concat(axis=2, values=[tf.reduce_max(matching_matrix, axis=2), tf.reduce_mean(matching_matrix,
                                                                                            axis=2)])  # [batch_size, passage_len, 2*decompse_dim]


def match_passage_with_question(x1, x2, x1_mask, x2_mask, scope="match_x_with_y", training=True):
    x_repre = tf.multiply(x1, tf.expand_dims(x1_mask, axis=-1))
    y_repre = tf.multiply(x2, tf.expand_dims(x2_mask, axis=-1))
    all_x_aware_y_representation = []

    with tf.variable_scope(scope or "match_x_with_y", reuse=tf.AUTO_REUSE):
        relevancy_matrix = cal_relevancy_matrix(x_repre, y_repre)
        relevancy_matrix = mask_relevancy_matrix(relevancy_matrix, x1_mask, x2_mask)
        all_x_aware_y_representation.append(tf.reduce_max(relevancy_matrix, axis=2, keepdims=True))
        all_x_aware_y_representation.append(tf.reduce_mean(relevancy_matrix, axis=2, keepdims=True))
        if hp.with_maxpool_match:
            maxpooling_decomp_params = tf.get_variable("maxpooling_decomp_params",
                                                       shape=[hp.cosine_MP_dim, hp.d_model], dtype=tf.float32)
            maxpooling_rep = cal_maxpooling_matching(x_repre, y_repre, maxpooling_decomp_params)
            all_x_aware_y_representation.append(maxpooling_rep)

        if hp.with_full_match:
            attentive_rep = multi_perspective_match(x_repre, y_repre, scope='mp_full_match')
            all_x_aware_y_representation.append(attentive_rep)

        if hp.with_attentive_match:
            atten_scores = calcuate_attention(x_repre, y_repre, hp.d_model, hp.d_model,
                    scope_name="attention", att_type=hp.att_type, att_dim=hp.att_dim,
                    remove_diagnoal=False, mask1=x1_mask, mask2=x2_mask, training=training)

            att_question_contexts = tf.matmul(atten_scores, x_repre)

            attentive_rep = multi_perspective_match(x_repre, att_question_contexts,
                                                    scope='mp_att_match')
            all_x_aware_y_representation.append(attentive_rep)

        if hp.with_max_attentive_match:
            max_att = cal_max_question_representation(x_repre, relevancy_matrix)
            max_attentive_rep = multi_perspective_match(x_repre, max_att,
                                                                     scope='mp_max_att')
            all_x_aware_y_representation.append(max_attentive_rep)


    all_x_aware_y_representation = tf.concat(axis=2, values=all_x_aware_y_representation)

    return all_x_aware_y_representation


def multi_perspective_match(repre1, repre2, scope='mp_match', reuse=tf.AUTO_REUSE):
    input_shape = tf.shape(repre1)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    matching_result = []

    with tf.variable_scope(scope, reuse=reuse):
        if hp.with_cosine:
            cosine_value = cosine_distance(repre1, repre2, cosine_norm=False)
            cosine_value = tf.reshape(cosine_value, [batch_size, seq_length, 1])
            matching_result.append(cosine_value)
        if hp.with_mp_cosine:
            mp_cosine_params = tf.get_variable("mp_cosine", shape=[hp.cosine_MP_dim, hp.d_model],
                                               dtype=tf.float32)
            mp_cosine_params = tf.expand_dims(mp_cosine_params, axis=0)
            mp_cosine_params = tf.expand_dims(mp_cosine_params, axis=0)

            repre1_flat = tf.expand_dims(repre1, axis=2)
            repre2_flat = tf.expand_dims(repre2, axis=2)
            # print(repre1_flat.shape)
            # print(mp_cosine_params.shape)
            mp_cosine_matching = cosine_distance(tf.multiply(repre1_flat, mp_cosine_params), repre2_flat)
            matching_result.append(mp_cosine_matching)
    matching_result = tf.concat(axis=2, values=matching_result)
    return matching_result