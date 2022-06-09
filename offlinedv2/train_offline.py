import collections
import functools
import logging
import os
import pathlib
import re
import warnings

try:
    import rich.traceback

    rich.traceback.install()
except ImportError:
    pass

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel('ERROR')
warnings.filterwarnings('ignore', '.*box bound precision lowered.*')

import numpy as np
import ruamel.yaml as yaml

from dreamerv2 import agent
from dreamerv2 import common


def make_env(mode, config, logdir):
    suite, task = config.task.split('_', 1)
    if suite == 'dmc':
        env = common.DMC(
            task, config.action_repeat, config.render_size, config.dmc_camera)
        env = common.NormalizeAction(env)
    elif suite == 'dmcmt':
        env = common.DMCMultitask(
            task, config.action_repeat, config.render_size, config.dmc_camera)
        env = common.NormalizeAction(env)
    elif suite == 'atari':
        env = common.Atari(
            task, config.action_repeat, config.render_size,
            config.atari_grayscale)
        env = common.OneHotAction(env)
    elif suite == 'crafter':
        assert config.action_repeat == 1
        outdir = logdir / 'crafter' if mode == 'train' else None
        reward = bool(['noreward', 'reward'].index(task)) or mode == 'eval'
        env = common.Crafter(outdir, reward)
        env = common.OneHotAction(env)
    else:
        raise NotImplementedError(suite)
    env = common.TimeLimit(env, config.time_limit)
    return env


def main():
    config_path = pathlib.Path('offlinedv2/dreamerv2') / 'configs.yaml'
    configs = yaml.safe_load(config_path.read_text())
    parsed, remaining = common.Flags(configs=['defaults']).parse(known_only=True)
    config = common.Config(configs['defaults'])
    for name in parsed.configs:
        config = config.update(configs[name])
    config = common.Flags(config).parse(remaining)

    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    config.save(logdir / 'config.yaml')
    print(config, '\n')
    print('Logdir', logdir)

    import tensorflow as tf
    tf.config.experimental_run_functions_eagerly(not config.jit)
    message = 'No GPU found. To actually train on CPU remove this assert.'
    assert tf.config.experimental.list_physical_devices('GPU'), message
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    assert config.precision in (16, 32), config.precision
    if config.precision == 16:
        from tensorflow.keras.mixed_precision import experimental as prec
        prec.set_policy(prec.Policy('mixed_float16'))

    print('Offline Datadir', config.offline_dir)
    train_replay = common.OfflineReplay(list(config.offline_dir), split_val=config.offline_split_val, **config.replay)
    eval_replay = common.Replay(
        logdir / 'eval_episodes',
        capacity=config.replay.capacity // 10,
        minlen=config.dataset.length,
        maxlen=config.dataset.length,
        disable_save=True,
    )
    step = common.Counter(0)
    outputs = [
        common.TerminalOutput(),
        common.JSONLOutput(logdir),
        common.TensorBoardOutput(logdir),
    ]
    logger = common.Logger(step, outputs)
    metrics = collections.defaultdict(list)

    should_video_eval = common.Every(config.eval_every)

    def per_episode(ep, mode):
        length = len(ep['reward']) - 1
        score = float(ep['reward'].astype(np.float64).sum())
        print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
        logger.scalar(f'{mode}_return', score)
        logger.scalar(f'{mode}_length', length)
        for key, value in ep.items():
            if re.match(config.log_keys_sum, key):
                logger.scalar(f'sum_{mode}_{key}', ep[key].sum())
            if re.match(config.log_keys_mean, key):
                logger.scalar(f'mean_{mode}_{key}', ep[key].mean())
            if re.match(config.log_keys_max, key):
                logger.scalar(f'max_{mode}_{key}', ep[key].max(0).mean())
        # if should_video_eval(step):
        #     for key in config.log_keys_video:
        #         logger.video(f'{mode}_policy_{key}', ep[key])
        replay = dict(train=train_replay, eval=eval_replay)[mode]
        logger.add(replay.stats, prefix=mode)
        logger.write()

    print('Create envs.')
    num_eval_envs = min(config.envs, config.eval_eps)
    if config.envs_parallel == 'none':
        eval_envs = [make_env('eval', config, logdir) for _ in range(num_eval_envs)]
    else:
        make_async_env = lambda mode: common.Async(
            functools.partial(make_env, mode, config, logdir), config.envs_parallel)
        eval_envs = [make_async_env('eval') for _ in range(num_eval_envs)]
    act_space = eval_envs[0].act_space
    obs_space = eval_envs[0].obs_space
    eval_driver = common.Driver(eval_envs)
    eval_driver.on_episode(lambda ep: per_episode(ep, mode='eval'))
    eval_driver.on_episode(eval_replay.add_episode)

    print(f'Precollecting eval steps.')
    random_agent = common.RandomAgent(act_space)
    eval_driver(random_agent, episodes=1)
    eval_driver.reset()

    print('Create agent.')
    train_dataset = iter(train_replay.dataset(**config.offline_model_dataset))
    if config.offline_split_val:
        val_dataset = iter(train_replay.dataset(validation=True, **config.offline_model_dataset))

    agnt = agent.Agent(config, obs_space, act_space, step)
    agnt.train(next(train_dataset))
    eval_policy = lambda *args: agnt.policy(*args, mode='eval')

    if config.offline_model_loaddir != 'none':
        print('Loading model', config.offline_model_loaddir)
        agnt.wm.load(config.offline_model_loaddir)
    else:
        print('Start model training.')
        model_step = common.Counter(0)
        model_logger = common.Logger(model_step, outputs)
        while model_step < config.offline_model_train_steps:
            if config.offline_split_val:
                # Compute model validation loss as average over chunks of data
                val_losses = []
                for _ in range(10):
                    val_loss, _, _, val_mets = agnt.wm.loss(next(val_dataset))
                    val_losses.append(val_loss)
                model_logger.scalar(f'validation_model_loss', np.array(val_losses, np.float64).mean())

            for it in range(100):
                model_step.increment()
                mets = agnt.model_train(next(train_dataset))

                for key, value in mets.items():
                    metrics[key].append(value)

            for name, values in metrics.items():
                model_logger.scalar(name, np.array(values, np.float64).mean())
                metrics[name].clear()
            model_logger.write()

            if model_step % config.offline_model_save_every == 0:
                print('Saving model')
                agnt.wm.save(logdir / f'model_{model_step.value}.pkl')
        agnt.wm.save(logdir / f'final_model.pkl')

    # Begin training
    metrics = collections.defaultdict(list)

    train_dataset = iter(train_replay.dataset(**config.offline_train_dataset))
    eval_dataset = iter(eval_replay.dataset(**config.offline_train_dataset))
    while step < config.steps:
        logger.write()
        print('Start evaluation.')
        if should_video_eval(step):
            logger.add(agnt.report(next(eval_dataset)), prefix='eval')
        eval_driver(eval_policy, episodes=config.eval_eps)

        print('Start training.')
        for it in range(200):
            step.increment()
            mets = agnt.agent_train(next(train_dataset))
            for key, value in mets.items():
                metrics[key].append(value)
        for name, values in metrics.items():
            logger.scalar(name, np.array(values, np.float64).mean())
            metrics[name].clear()

        logger.write(fps=True)
        agnt.save(logdir / 'variables.pkl')

    for env in eval_envs:
        try:
            env.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()
