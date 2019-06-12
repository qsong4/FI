import argparse

class Hparams:
    parser = argparse.ArgumentParser()

    # prepro
    parser.add_argument('--vocab_size', default=32000, type=int)

    # train
    ## files
    parser.add_argument('--train', default='./data/train.tsv',
                             help="training data")
    parser.add_argument('--eval', default='./data/dev.tsv',
                             help="evaluation data")

    ## vocabulary
    parser.add_argument('--vocab', default='./data/esb.vocab',
                        help="vocabulary file path")

    # training scheme
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--eval_batch_size', default=128, type=int)

    parser.add_argument('--lr', default=0.0003, type=float, help="learning rate")
    parser.add_argument('--warmup_steps', default=4000, type=int)
    parser.add_argument('--num_epochs', default=20, type=int)

    # model
    parser.add_argument('--d_model', default=512, type=int,
                        help="hidden dimension of interativate")
    parser.add_argument('--d_ff', default=2048, type=int,
                        help="hidden dimension of feedforward layer")
    parser.add_argument('--num_blocks', default=6, type=int,
                        help="number of encoder/decoder blocks")
    parser.add_argument('--num_heads', default=8, type=int,
                        help="number of attention heads")
    parser.add_argument('--maxlen', default=100, type=int,
                        help="maximum length of a source sequence")
    parser.add_argument('--num_class', default=2, type=int,
                        help="number of class")
    parser.add_argument('--dropout_rate', default=0.3, type=float)

    # test
