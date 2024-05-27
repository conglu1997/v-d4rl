# V-D4RL

[![Twitter](https://badgen.net/badge/icon/twitter?icon=twitter&label)](https://twitter.com/cong_ml/status/1536352379242676228)
[![arXiv](https://img.shields.io/badge/arXiv-2206.04779-b31b1b.svg)](https://arxiv.org/abs/2206.04779)

<p align="center">
  <img src="figs/envs.png" />
</p>

V-D4RL provides pixel-based analogues of the popular D4RL benchmarking tasks, derived from the **`dm_control`** suite, along with natural extensions of two state-of-the-art online pixel-based continuous control algorithms, DrQ-v2 and DreamerV2, to the offline setting. For further details, please see the paper:

**_Challenges and Opportunities in Offline Reinforcement Learning from Visual Observations_**; Cong Lu*, Philip J. Ball*, Tim G. J. Rudner, Jack Parker-Holder, Michael A. Osborne, Yee Whye Teh. Published at [TMLR, 2023](https://openreview.net/forum?id=1QqIfGZOWu).

<p align="center">
  <a href=https://arxiv.org/abs/2206.04779>View on arXiv</a>
</p>

## Benchmarks
The V-D4RL datasets can be found on [Google Drive](https://drive.google.com/drive/folders/15HpW6nlJexJP5A4ygGk-1plqt9XdcWGI?usp=sharing)[^1]. **These must be downloaded before running the code.** Assuming the data is stored under `vd4rl_data`, the file structure is:

```
vd4rl_data
└───main
│   └───walker_walk
│   │   └───random
│   │   │   └───64px
│   │   │   └───84px
│   │   └───medium_replay
│   │   │   ...
│   └───cheetah_run
│   │   ...
│   └───humanoid_walk
│   │   ...
└───distracting
│   ...
└───multitask
│   ...
```

A complete listing of the datasets in `main` is given in Table 5, Appendix A.

**(Update, Dec 2023):** Our datasets are now compatible with [torch/rl](https://github.com/pytorch/rl/pull/1756)!

## Baselines

### Environment Setup
Requirements are presented in conda environment files named `conda_env.yml` within each folder. The command to create the environment is:
```
conda env create -f conda_env.yml
```

Alternatively, dockerfiles are located under `dockerfiles`, replace `<<USER_ID>>` in the files with your own user ID from the command `id -u`.

### V-D4RL Main Evaluation
Example run commands are given below, given an environment type and dataset identifier:

```
ENVNAME=walker_walk # choice in ['walker_walk', 'cheetah_run', 'humanoid_walk']
TYPE=random # choice in ['random', 'medium_replay', 'medium', 'medium_expert', 'expert']
```

#### Offline DV2 
```
python offlinedv2/train_offline.py --configs dmc_vision --task dmc_${ENVNAME} --offline_dir vd4rl_data/main/${ENV_NAME}/${TYPE}/64px --offline_penalty_type meandis --offline_lmbd 10 --seed 0
```

#### DrQ+BC
```
python drqbc/train.py task_name=offline_${ENVNAME}_${TYPE} offline_dir=vd4rl_data/main/${ENV_NAME}/${TYPE}/84px nstep=3 seed=0
```

#### DrQ+CQL
```
python drqbc/train.py task_name=offline_${ENVNAME}_${TYPE} offline_dir=vd4rl_data/main/${ENV_NAME}/${TYPE}/84px algo=cql cql_importance_sample=false min_q_weight=10 seed=0
```

#### BC
```
python drqbc/train.py task_name=offline_${ENVNAME}_${TYPE} offline_dir=vd4rl_data/main/${ENV_NAME}/${TYPE}/84px algo=bc seed=0
```

### Distracted and Multitask Experiments
To run the distracted and multitask experiments, it suffices to change the offline directory passed to the commands above.

## Note on data collection and format
We follow the image sizes and dataset format of each algorithm's native codebase.
The means that Offline DV2 uses `*.npz` files with 64px images to store the offline data, whereas DrQ+BC uses `*.hdf5` with 84px images.

The data collection procedure is detailed in Appendix B of our paper, and we provide conversion scripts in `conversion_scripts`.
For the original SAC policies to generate the data see [here](https://github.com/philipjball/SAC_PyTorch/blob/dmc_branch/train_agent.py).
See [here](https://github.com/philipjball/SAC_PyTorch/blob/dmc_branch/gather_offline_data.py) for distracted/multitask variants.
We used `seed=0` for all data generation.

## Acknowledgements
V-D4RL builds upon many works and open-source codebases in both offline reinforcement learning and online pixel-based continuous control. We would like to particularly thank the authors of:
- [D4RL](https://github.com/rail-berkeley/d4rl)
- [DMControl](https://github.com/deepmind/dm_control)
- [DreamerV2](https://github.com/danijar/dreamerv2)
- [DrQ-v2](https://github.com/facebookresearch/drqv2)
- [LOMPO](https://github.com/rmrafailov/LOMPO)

## Contact
Please contact [Cong Lu](mailto:conglu97*AT*outlook*DOT*com) or [Philip Ball](mailto:ball*AT*robots*DOT*ox*DOT*ac*DOT*uk) for any queries.
We welcome any suggestions or contributions! 


[^1]: The files on Google Drive may be separated into separate zip files, use [this](https://stackoverflow.com/questions/60842075/combine-the-split-zip-files-downloading-from-google-drive) to combine.
