import argparse

from preprocessing import preprocessing
from models.rnn.test import test
from models.rnn.train_v import train

def main(args):
    if args.preprocessing:
        preprocessing(args)
    elif args.train:
        train(args)
    elif args.train_v:
        train(args)
    elif args.test:
        test(args)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessing', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--train_v', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--variational', action='store_true')

    # Path
    parser.add_argument('--data_path', default='/HDD/yehoon/data', type=str)




    args = parser.parse_args()
    main(args)
