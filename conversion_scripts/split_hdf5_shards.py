import pathlib
import numpy as np
import h5py
import argparse


def main():
    # Include argument parser
    parser = argparse.ArgumentParser(description='Split hdf5 shards to smaller.')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to input files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to output files')
    parser.add_argument('--split_size', type=int, default=5)
    args = parser.parse_args()

    in_dir = pathlib.Path(args.input_dir)
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counter = 0
    for filename in in_dir.glob('*.hdf5'):
        print(filename)
        with h5py.File(filename, "r") as episodes:
            episodes = {k: episodes[k][:] for k in episodes.keys()}
            print(episodes['action'].shape[0])

            split_episodes = {k: np.array_split(v, args.split_size) for k, v in episodes.items()}
            split_episodes = [{k: v[idx] for k, v in split_episodes.items()} for idx in range(args.split_size)]

            for eps in split_episodes:
                with h5py.File(out_dir / f'{counter}.hdf5', 'w') as shard_file:
                    for k, v in eps.items():
                        shard_file.create_dataset(k, data=v, compression='gzip')
                print(counter)
                counter += 1


if __name__ == '__main__':
    main()
