import argparse
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('policy', help='policy file')
    args = parser.parse_args()
    with open(args.policy, 'rb') as f:
        policy = pickle.load(f)
    print(policy)


if __name__ == '__main__':
    main()
