from common import parser, training_args, train
if __name__=='__main__':
    train(training_args(parser('Train a BayesFlow parameter posterior',model=True)).parse_args())
