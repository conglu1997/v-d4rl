import pathlib
import numpy as np
import h5py
import cv2
import argparse


def load_episodes(directory, capacity=None):
    # The returned directory from filenames to episodes is guaranteed to be in
    # temporally sorted order.
    filenames = sorted(directory.glob('*.npz'))
    if capacity:
        num_steps = 0
        num_episodes = 0
        for filename in reversed(filenames):
            length = int(str(filename).split('-')[-1][:-4])
            num_steps += length
            num_episodes += 1
            if num_steps >= capacity:
                break
        filenames = filenames[-num_episodes:]
    episodes = {}
    for filename in filenames:
        try:
            with filename.open('rb') as f:
                episode = np.load(f)
                episode = {k: episode[k] for k in episode.keys()}
                # Conversion for older versions of npz files.
                if 'is_terminal' not in episode:
                    episode['is_terminal'] = episode['discount'] == 0.
        except Exception as e:
            print(f'Could not load episode {str(filename)}: {e}')
            continue
        episodes[str(filename)] = episode
    return episodes


def main():
    # Include argument parser
    parser = argparse.ArgumentParser(description='Convert npz files to hdf5.')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to input files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to output files')
    args = parser.parse_args()

    step_type = np.ones(501)
    step_type[0] = 0
    step_type[500] = 2

    output = {}
    episodes = load_episodes(pathlib.Path(args.input_dir))
    episodes = list(episodes.values())

    actions = [e['action'] for e in episodes]
    discounts = [e['discount'] for e in episodes]
    observations = []
    for e in episodes:
        resized_images = np.empty((501, 84, 84, 3), dtype=e['image'].dtype)
        for (k, i) in enumerate(e['image']):
            resized_images[k] = cv2.resize(i, dsize=(84, 84), interpolation=cv2.INTER_CUBIC)
        observations.append(resized_images.transpose(0, 3, 1, 2))
    rewards = [e['reward'] for e in episodes]
    step_types = [step_type for _ in episodes]

    output['action'] = np.concatenate(actions)
    output['discount'] = np.concatenate(discounts)
    output['observation'] = np.concatenate(observations)
    output['reward'] = np.concatenate(rewards)
    output['step_type'] = np.concatenate(step_types)

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_dir / 'data.hdf5', 'w') as shard_file:
        for k, v in output.items():
            shard_file.create_dataset(k, data=v, compression='gzip')


if __name__ == '__main__':
    main()
