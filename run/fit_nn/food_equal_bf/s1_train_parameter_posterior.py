from _shared.common import parser, training_args, train


def main(*, unified=False):
    p = parser('Train a BayesFlow parameter posterior', model=True, unified=unified)
    train(training_args(p).parse_args())


if __name__ == '__main__':
    main(unified=True)
