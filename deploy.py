"""Usage: python deploy.py
    [--mode STR]
    [--cluster_name=<cluster name used by train.py, to speed things up>]
    --best-checkpoint-path=<checkpoint path returned by >

Example:

    python deploy.py --mode job --checkpoints-dir s3://<bucket>/<dir>/
        --best-checkpoint-path <path to a local checkpoint dir>

    python deploy.py --mode job --checkpoints-dir gs://<bucket>/<dir>/
        --best-checkpoint-path <path to a local checkpoint dir>

    python deploy.py --mode job --checkpoints-dir gs://<bucket>/<dir>/ --epoch <epoch>
"""
import os
import argparse
from datetime import datetime
from filelock import FileLock
import json
import logging
import requests
import subprocess
import time
from urllib import parse

import ray
from ray.rllib.agents import ppo
from ray.rllib.env.env_context import EnvContext
from ray import serve

import env
import train


RUN_PREFIX = "TUNE-RL-SERVE"


def sync_chkpt(ckpt_path, bucket):
    print("Starting checkpoint sync...")

    ts = datetime.now().strftime('%Y%m%d-%H%M')
    model_path = os.path.expanduser(f"~/ray_model_{ts}")

    remote_ckpt_dir, ckpt_name = os.path.split(ckpt_path)
    local_ckpt_path = os.path.join(model_path, ckpt_name)

    if os.path.exists(local_ckpt_path):
        # Already synced, probably by other serving replicas.
        print("Model checkpoint already exists, skip syncing. ",
              local_ckpt_path)
        return local_ckpt_path

    if bucket.startswith("gs"):
        sync = ["gsutil", "rsync"]
    elif bucket.startswith("s3"):
        sync = ["aws", "s3", "sync"]
    subprocess.run(["mkdir", "-p", model_path], check=True)
    subprocess.run(sync + [remote_ckpt_dir, model_path], check=True)

    print("Synced", ckpt_path, "to", model_path)

    return local_ckpt_path


@serve.deployment(name="corridor", num_replicas=2)
class Corridor(object):
    def __init__(self, ckpt_path, local, bucket):
        print("Deployment initializing.", locals())
        config = ppo.DEFAULT_CONFIG.copy()
        config.update(train.TRAINER_CFG)
        config['num_workers'] = 0

        if local:
            model_path = ckpt_path
        else:
            assert bucket, "Bucket must be provided if not local."
            with FileLock("/tmp/checkpoint.lock"):
                model_path = sync_chkpt(ckpt_path, bucket)
        agent = ppo.PPOTrainer(config=config, env=env.MysteriousCorridor)
        agent.restore(model_path)
        print("Agent restored")

        self._policy = agent.workers.local_worker().get_policy()
        self._count = 0

    def __action(self, state):
        action = self._policy.compute_single_action(state)
        # JSON can't handle int64. Convert to int32.
        return int(action[0])

    async def __call__(self, request):
        self._count += 1

        body = await request.body()
        try:
            data = json.loads(body.decode("utf-8"))
        except ValueError as e:
            # Can't parse body as json data.
            return "can't decode: " + body.decode("utf-8")

        try:
            return {
                "count": self._count,
                "action": self.__action(data["state"]),
            }
        except Exception as e:
            return str(e)


def deploy(ckpt_path: str, local: bool, bucket: str = None):
    serve.start(detached=True)
    Corridor.deploy(ckpt_path, local=local, bucket=bucket)

    print("Corridor service deployed!")
    print(f"You can query the model at: {Corridor.url}")
    return Corridor.url


def init_ray(cluster_name: str, mode: str):
    starttime = time.time()
    if mode in ("local", "job"):
        assert not cluster_name, f"No need to provide a cluster name for mode {mode}."
        ray.init(address="auto")
    elif cluster_name:
        ray.init(address=f"anyscale://{cluster_name}")
    else:
        ray.init(
            # the address of the Ray cluster.
            # In Anyscale, you don't need an IP like open-source Ray
            # just call a cluster whatever you like and it will be created
            # or re-used (if already exists)
            address=f"anyscale://my-serve-cluster-{int(starttime)}",

            # this will upload this directory to Anyscale so that
            # the code can be run on cluster
            project_dir=".",

            # Install all the dependencies.
            runtime_env={"pip": "./requirements.txt"}
        )

    print(f'Cluster launch took {time.time() - starttime:.2f} seconds...')


def best_checkpoint(args):
    if args.best_checkpoint_path:
        return args.best_checkpoint_path
    if args.checkpoints_dir and args.epoch:
        best_checkpoint_file = os.path.join(
            args.checkpoints_dir, f"{RUN_PREFIX}-{args.epoch}", "best_checkpoint.txt")
        tmp_file = "/tmp/best_checkpoint.txt"
        if args.checkpoints_dir.startswith("gs"):
            cp = ["gsutil", "cp"]
        elif args.checkpoints_dir.startswith("s3"):
            cp = ["aws", "s3", "cp"]
        subprocess.run(cp + [best_checkpoint_file, tmp_file], check=True)

        with open(tmp_file, "r") as f:
            return f.read().strip()
    return ""


def def_flags(parser):
    parser.add_argument(
        "--best-checkpoint-path",
        type=str,
        default="",
        help="Path of an RLlib checkpoint to serve.")
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        help="Path to the cloud bucket where the checkpoint is stored.")
    parser.add_argument(
        "--epoch",
        help="Epoch for the training run, whose result we are trying to serve.")
    parser.add_argument(
        "--cluster-name",
        type=str,
        default="",
        help="Existing session to speed up deployment.")
    parser.add_argument(
        "--mode",
        choices=("local", "job", "client"),
        help="Whether this is running in local mode, in a job/head node, or via client.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    def_flags(parser)
    args = parser.parse_args()

    assert args.checkpoints_dir, "Must specify --checkpoints-dir."

    ckpt_path = best_checkpoint(args)
    assert ckpt_path, "Must specify --best-checkpoint-path or --epoch."

    init_ray(args.cluster_name, args.mode)

    serve_url = deploy(ckpt_path, local=(args.mode == "local"), bucket=args.checkpoints_dir)

    if args.mode == "job":
        time.sleep(1000)
    else:
        from test_deployment_locally import test_deployment
        test_deployment(serve_url)
        print("serve shutting down")
        serve.shutdown()
        ray.shutdown()
