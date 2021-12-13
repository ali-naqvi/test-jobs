"""Usage: python train.py [--mode STR] [--checkpoints-dir STR]

Example:

    python train.py --mode job --checkpoints-dir s3
    python train.py --mode local

Example with jobs:

    ray job submit --runtime-env ./renv.yaml -- python train.py --mode job --checkpoints-dir s3://<bucket>/<dir>/
    ray job submit --runtime-env ./renv.yaml -- python train.py --mode job --checkpoints-dir gs://<bucket>/<dir>/
    ray job submit --runtime-env ./renv.yaml -- python train.py --mode job --checkpoints-dir gs://<bucket>/<dir>/ --epoch <epoch>
"""

import argparse
from datetime import datetime
import os
import subprocess
import time
from typing import Any, Dict, Tuple

import ray
from ray import tune
from ray.rllib.agents.ppo import ppo

import env
import model


TRAINER_CFG = {
    "env": env.MysteriousCorridor,
    "env_config": {
        "reward": 10.0,
    },
    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
    "model": {
        "custom_model": model.TorchCustomModel,
        "fcnet_hiddens": [20, 20],
        "vf_share_layers": True,
    },
    "num_workers": 1,  # parallelism
    "framework": "torch",
    "rollout_fragment_length": 10,
    "lr": 0.01,
}

RUN_PREFIX = "TUNE-RL-SERVE"


def train(ts: str, mode: str, upload_dir: str) -> str:
    print("Training & tuning automatically with Ray Tune...")
    local = mode == "local"

    run_name = f"{RUN_PREFIX}-{ts}"
    results = tune.run(
        ppo.PPOTrainer,
        name=run_name,
        config=TRAINER_CFG,
        checkpoint_freq=5,
        checkpoint_at_end=True,
        sync_config=None if local else tune.SyncConfig(upload_dir=upload_dir),
        stop={"training_iteration": 10},
        num_samples=1 if local else 10,
        metric="episode_reward_mean",
        mode="max")

    print("Best checkpoint: ")
    print(results.best_checkpoint)

    if upload_dir:
        tmp_file = "/tmp/best_checkpoint.txt"
        with open(tmp_file, "w") as f:
            f.write(results.best_checkpoint)
        best_checkpoint_file = os.path.join(
            upload_dir, run_name, "best_checkpoint.txt")
        print("Saving best checkpoint in: ", best_checkpoint_file)

        if upload_dir.startswith("gs://"):
            subprocess.run(["gsutil", "cp", tmp_file, best_checkpoint_file],
                           check=True)
        elif upload_dir.startswith("s3://"):
            subprocess.run(["aws", "s3", "cp", tmp_file, best_checkpoint_file],
                           check=True)
        else:
            raise ValueError("Unknown upload dir type: ", upload_dir)

    return results.best_checkpoint


def def_flags(parser):
    parser.add_argument(
        "--epoch",
        help="Epoch for this training run.")
    parser.add_argument(
        "--checkpoints-dir",
        help="Directory for Ray Tune to upload model checkponts to.")
    parser.add_argument(
        "--mode",
        choices=("local", "job", "client"),
        help=("Whether this is running in local mode, "
              "in a job/head node, or via client."))


def init_ray(ts: str, mode: str):
    if mode in ("local", "job"):
        ray.init(address="auto")
    else:
        starttime = time.time()
        cluster_name = f"my-cluster-{ts}"
        ray.init(
            # the address of the Ray cluster.
            # In Anyscale, you don't need an IP like open-source Ray
            # just call a cluster whatever you like and it will be created
            # or re-used (if already exists)
            address=f"anyscale://{cluster_name}",

            # this will upload this directory to Anyscale so that
            # the code can be run on cluster
            project_dir=".",

            # Install all the dependencies.
            runtime_env={"pip": "./requirements.txt"}
        )

        print(f'Took {time.time() - starttime:.2f} seconds '
              'to launch cluster {cluster_name}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    def_flags(parser)
    args = parser.parse_args()

    print(f"Running with following CLI options: {args}")

    starttime = time.time()

    ts = args.epoch if args.epoch else datetime.now().strftime('%Y%m%d-%H%M')
    init_ray(ts, args.mode)
    train(ts, args.mode, upload_dir=args.checkpoints_dir)

    print(f'Training took {time.time() - starttime:.2f} seconds...')
