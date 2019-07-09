import argparse

class Hparams:
    parser = argparse.ArgumentParser()

    # prepro
    parser.add_argument('--vocab_size', default=32000, type=int)

    # train
    ## files
    parser.add_argument('--train', default='./data/snli_train.tsv',
                             help="training data")
    parser.add_argument('--eval', default='./data/snli_test.tsv',
                             help="evaluation data")

    parser.add_argument('--model_path', default='FImatchE%02dL%.2fA%.2f')
    parser.add_argument('--modeldir', default='./model')
    parser.add_argument('--vec_path', default='./data/vec/snil_trimmed_vec.npy')

    ## vocabulary
    parser.add_argument('--vocab', default='./data/snli.vocab',
                        help="vocabulary file path")

    # training scheme
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--eval_batch_size', default=128, type=int)
    parser.add_argument('--preembedding', default=False, type=bool)
    parser.add_argument('--early_stop', default=3, type=int)

    #learning rate 0.0003 is too high
    parser.add_argument('--lr', default=0.0003, type=float, help="learning rate")
    parser.add_argument('--warmup_steps', default=4000, type=int)
    parser.add_argument('--num_epochs', default=20, type=int)
    parser.add_argument('--lambda_l2', default=0.004, type=float)

    # match
    parser.add_argument('--with_maxpool_match', default=False, type=bool)
    parser.add_argument('--with_full_match', default=True, type=bool)
    parser.add_argument('--with_attentive_match', default=False, type=bool)
    parser.add_argument('--with_max_attentive_match', default=False, type=bool)
    parser.add_argument('--with_inference', default=True, type=bool)
    parser.add_argument('--att_dim', default=50, type=int)
    parser.add_argument('--att_type', default="symmetric")

    #cnn agg
    parser.add_argument('--filter_sizes', default='3,4,5')
    parser.add_argument('--num_filters', default=128, type=int)

    # model
    # This is also the word embedding size , and must can divide by head num.
    parser.add_argument('--cosine_MP_dim', default=20, type=int,
                        help="cosine_MP_dim")# tips MP_dim+1 should be abled divide by num_heads
    parser.add_argument('--with_cosine', default=True, type=bool,
                        help="with_cosine")
    parser.add_argument('--with_mp_cosine', default=True, type=bool,
                        help="with_mp_cosine")
    parser.add_argument('--d_model', default=300, type=int,
                        help="hidden dimension of interativate")
    parser.add_argument('--d_ff', default=512, type=int,
                        help="hidden dimension of feedforward layer")
    parser.add_argument('--num_extract_blocks', default=3, type=int,
                        help="number of extract blocks")
    parser.add_argument('--num_inter_blocks', default=3, type=int,
                        help="number of inter blocks")
    parser.add_argument('--num_agg_blocks', default=3, type=int,
                        help="number of agg blocks")
    parser.add_argument('--num_heads', default=6, type=int,
                        help="number of attention heads")
    parser.add_argument('--maxlen', default=50, type=int,
                        help="maximum length of a source sequence")
    parser.add_argument('--num_class', default=3, type=int,
                        help="number of class")
    parser.add_argument('--dropout_rate', default=0.1, type=float)
    parser.add_argument('--is_training', default=True, type=bool)

    # test
    parser.add_argument('--test_file', default='./data/snli_test.tsv')
