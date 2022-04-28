import argparse

from preprocessing import preprocessing
from train import train

def main(args):
    if args.preprocessing:
        preprocessing()
    elif args.train:
        train()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessing', action='store_true')
    parser.add_argument('--train', action='store_true')


    args = parser.parse_args()
    main(args)