import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("npy", help='path to the input npy matrix')
    parser.add_argument("csv", help='path to the csv to be dumped')
    args = parser.parse_args()

    x = np.load(args.npy)
    np.savetxt(args.csv, x, fmt='%.6f', delimiter=',')