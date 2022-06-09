import datetime
import io
import pathlib
import uuid
import numpy as np
import h5py
import argparse


def save_episode(directory, episode):
    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    identifier = str(uuid.uuid4().hex)
    length = len(episode['action'])
    filename = directory / f'{timestamp}-{identifier}-{length}.npz'
    with io.BytesIO() as f1:
        np.savez_compressed(f1, **episode)
        f1.seek(0)
        with filename.open('wb') as f2:
            f2.write(f1.read())
    return filename


def main():
    # Include argument parser
    parser = argparse.ArgumentParser(description='Convert hdf5 files to npz.')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to input files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to output files')
    args = parser.parse_args()

    is_last = np.zeros(501, dtype=bool)
    is_last[500] = True
    is_first = np.zeros(501, dtype=bool)
    is_first[0] = True
    is_terminal = np.zeros(501, dtype=bool)

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    filenames = sorted(pathlib.Path(args.input_dir).glob('*.hdf5'))
    for filename in filenames:
        with h5py.File(filename, "r") as f:
            actions = f['action'][:]
            observations = f['observation'][:]
            rewards = f['reward'][:]
            discounts = f['discount'][:]
            while len(actions) > 0:
                ep = {
                    'image': observations[:501].transpose(0, 2, 3, 1),
                    'action': actions[:501],
                    'reward': rewards[:501],
                    'discount': discounts[:501],
                    'is_last': is_last,
                    'is_first': is_first,
                    'is_terminal': is_terminal,
                }
                actions = actions[501:]
                observations = observations[501:]
                rewards = rewards[501:]
                discounts = discounts[501:]
                save_episode(out_dir, ep)


if __name__ == '__main__':
    main()
