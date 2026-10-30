from common import parser, training_args, train
if __name__=='__main__':
    train(training_args(parser('Train the eight-model BayesFlow classifier')).parse_args(),classifier=True)
