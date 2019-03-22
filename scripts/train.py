import os
from os.path import join, basename, dirname
import sys
sys.path.append(join(dirname(__file__), '..'))
import argparse

from scanoise.training import train
from scanoise.utils import replace_dots

MODEL_DATA_MAP = {
    'dpav4': 'DPAv4',
    'shivam': 'Shivam',
    'random': 'Random',
    'ascad': 'ASCAD',
    'Benadjila': 'ASCAD',
    'Benadjila_dpav4': 'DPAv4',
    'Benadjila_shivam': 'Shivam',
    'Benadjila_random': 'Random',

}

def get_filenames(data_root, model_id):
    data_id = MODEL_DATA_MAP[model_id]
    return {
        'trace': join(data_root, '{}_traces.npy'.format(data_id)),
        'value': join(data_root, '{}_values.npy'.format(data_id))
    }

def main(dataroot='./data/datasets/', outroot='./data/results/',
         model_id='dpav4', n_trains='full', noise=0.5, fold=None, n_epochs=100):
    """"""

    subroot = (
        '{}/n{:05d}/noise{:.1f}/'
        if n_trains != 'full' else
        '{}/n{}/noise{:.1f}/'
    ).format(model_id, n_trains, noise)

    if fold and isinstance(fold, (int, float)):
        subroot = join(subroot, '{:02d}/'.format(fold))

    train(
        trace_fn = get_filenames(dataroot, model_id)['trace'],
        label_fn = get_filenames(dataroot, model_id)['value'],
        model_id = model_id,
        n_trains = n_trains,
        noise    = noise,
        n_epochs = n_epochs,
        out_root = join(outroot, replace_dots(subroot))
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", help='path to the root of feature files')

    parser.add_argument("result_path", help='path to the result report to be dumped')

    parser.add_argument("model_id", choices=set(MODEL_DATA_MAP.keys()),
                        help="model type", type=str, default='dpav4')

    parser.add_argument("n_trains", type=str, default='full',
                        help="number of traces to use in the training")

    parser.add_argument("noise", type=float, default=0.5,
                        help="amount of the noise to add in the input signal during training")

    parser.add_argument("--n-epochs", type=str, default=300,
                        help="number of epochs")

    parser.add_argument("--fold", type=int, default=0,
                        help="fold number (optional)")

    args = parser.parse_args()

    if args.n_trains != 'full':
        n_trains = int(args.n_trains)
    else:
        n_trains = args.n_trains

    # train!
    main(dataroot=args.dataset_path, outroot=args.result_path,
         model_id=args.model_id, n_trains=n_trains, noise=args.noise,
         n_epochs=args.n_epochs, fold=args.fold)
