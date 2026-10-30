from _shared.common import parser, training_args, train

if __name__ == '__main__':
    train(training_args(parser('Train a BayesFlow classifier for 2-8 selected models',
                               classifier=True)).parse_args(), classifier=True)
