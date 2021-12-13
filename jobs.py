"""Example Usage:

   ANYSCALE_HOST=<host> ANYSCALE_CLI_TOKEN=<token> python jobs.py \
       --cloud-name=<anyscale cloud name> \
       --checkpoints-dir=<path to cloud bucket> \
       --code-path=<path to working dir zip file>

   if cloud provider is GCP, please also specify --gcp-proj-id flag.
"""

import argparse
from datetime import datetime
import os
import requests
import time
from urllib import parse

from anyscale.sdk.anyscale_client.models.create_cluster_environment \
    import CreateClusterEnvironment
from anyscale.sdk.anyscale_client.models.create_cluster_compute \
    import CreateClusterCompute
from anyscale.sdk.anyscale_client.models.create_production_job \
    import CreateProductionJob
from anyscale.sdk.anyscale_client import (
    AWSNodeOptions,
    IamInstanceProfileSpecification
)
from anyscale import AnyscaleSDK

import env


CONSOLE_HOST = "https://console.anyscale.com"

CLUSTER_ENV_JSON = {
    "base_image": "anyscale/ray-ml:latest-py37-gpu",
    "python": {
        "pip_packages": [
            "tblib",
            "fastapi",
            "uvicorn",
            "anyscale",
            "requests",
            "torch",
            "gsutil",
            "awscli",
            "google-cloud-storage",
            "ray[all]",
        ],
    },
    "env_vars": {
        "GOOGLE_CLOUD_PROJECT": None,
    },
}


def build_cfgs(args: argparse.Namespace, epoch: str):
    cloud = sdk.search_clouds(clouds_query={
        "name": {"equals": args.cloud_name}
    }).results[0]

    cluster_env_name = f"my-cluster-env-{epoch}"
    print("Building cluster env ... ", cluster_env_name)
    print("Find your new build here: "
          f"{CONSOLE_HOST}/configurations/?state=CreatedByMe&tab=cluster-env")
    if cloud.provider == "GCP":
        if args.gcp_proj_id:
            CLUSTER_ENV_JSON["env_vars"]["GOOGLE_CLOUD_PROJECT"] = args.gcp_proj_id
        else:
            raise ValueError("Please specify GCP project id via --gcp-proj-id flag.")
    cluster_env = sdk.build_cluster_environment(
        create_cluster_environment=CreateClusterEnvironment(
            name=cluster_env_name,
            config_json=CLUSTER_ENV_JSON))
    print("Done, cluster environment ID: ", cluster_env.id)
    # cluster_env = None

    cluster_compute_name = f"my-cluster-compute-{epoch}"
    print("Creating compute config ... ", cluster_compute_name)
    print("Find your new compute config here: "
          f"{CONSOLE_HOST}/configurations/?state=CreatedByMe&tab=cluster-compute")
    default_compute_config = sdk.get_default_compute_config(cloud.id).result

    if cloud.provider == "AWS" and args.aws_role_arn:
        # we need to give the data plane account access to our S3 working directory
        aws_node_opts = AWSNodeOptions()
        iam_role = IamInstanceProfileSpecification(arn=args.aws_role_arn)
        aws_node_opts.iam_instance_profile = iam_role
        default_compute_config.aws = aws_node_opts

    cluster_compute = sdk.create_cluster_compute(
        create_cluster_compute=CreateClusterCompute(
            name=cluster_compute_name,
            config=default_compute_config)).result
    print("Done, cluster compute ID: ", cluster_compute.id)

    return cluster_env, cluster_compute


def _wait_for_status(job_id: str, goal_state: str) -> bool:
    while True:
        result = sdk.get_production_job(production_job_id=job_id).result
        state = result.state.current_state

        if state == "OUT_OF_RETRIES":
            return False
        elif state == goal_state:
            return True

        time.sleep(1)


def submit_train_job(args: argparse.Namespace, cluster_env: str,
                     cluster_compute: str, epoch: str):
    print("Submitting training job ...")

    default_proj = sdk.get_default_project().result

    cfg = {
        "build_id": cluster_env.id,
        "compute_config_id": cluster_compute.id,
        "runtime_env": {
            "working_dir": args.code_path,
        },
        "entrypoint": ("python train.py --mode job "
                       f"  --checkpoints-dir {args.checkpoints_dir} --epoch {epoch}"),
        "is_service": False,
        "max_retries": 3,
    }
    job = sdk.create_job(CreateProductionJob(
        project_id=default_proj.id,
        name=f"train_job_{epoch}",
        config=cfg,
    )).result

    print("Waiting for train job to finish ...")
    print(f"Find your train job here: {CONSOLE_HOST}/jobs/{job.id}")

    assert _wait_for_status(job.id, "SUCCESS"), ("Train job failed! Check the logs for errors")


def submit_deploy_job(args: argparse.Namespace, cluster_env: str,
                      cluster_compute: str, epoch: str) -> str:
    print("Submitting Ray Serve job ...")

    default_proj = sdk.get_default_project().result

    cfg = {
        "build_id": cluster_env.id,
        "compute_config_id": cluster_compute.id,
        "runtime_env": {
            "working_dir": args.code_path,
        },
        "entrypoint": ("python deploy.py --mode job "
                       f"  --checkpoints-dir {args.checkpoints_dir} --epoch {epoch} "
                       "&& sleep infinity"),
        "is_service": False,
        "max_retries": 3,
    }

    serve_job_name = f"serve_job_{epoch}"
    job = sdk.create_job(CreateProductionJob(
        project_id=default_proj.id,
        name=serve_job_name,
        config=cfg,
    )).result

    print("Waiting for serve job to start ...")
    print(f"Find your serve job here: {CONSOLE_HOST}/jobs/{job.id}")

    assert _wait_for_status(job.id, "RUNNING"), "Serve job failed! Check the logs for errors."

    return job.id


def test_deployment(url, token, iterations=20, max_retries=10):
    def get_action(url, state, token=None):
        cookies = {"anyscale-token": token} if token else {}

        sess = requests.Session()
        data = {"state": state}
        resp = sess.post(url, json=data, cookies=cookies)

        return resp.json()["action"]

    def render(env):
        print(env.render(), end = "\r")


    # Do a few retries util the Serve instance is available on the network.
    retries = 0
    while True:
        try:
            _ = get_action(url, state=0, token=token)
            # Successfully fetch action. Deployment is up.
            # Break so we can actually run test.
            break
        except Exception:
            retries += 1
            if retries > max_retries:
                print("Failed, endpoint is not up :(")
                return None
            print(f"Waiting for endpoint networking for {2**retries} seconds...")

            time.sleep(2**retries)

    test_env = env.MysteriousCorridor({"reward": 1.0})

    for _ in range(iterations):
        state = test_env.reset()
        done = False
        render(test_env)
        while not done:
            time.sleep(0.25)
            action = get_action(url, state[0], token)
            state, _, done, _ = test_env.step(action)
            render(test_env)
        time.sleep(0.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cloud-name",
        help="Name of the cloud to use. { aws_managed | aws | gcp }")
    parser.add_argument(
        "--gcp-proj-id",
        help="ID of the project if using a GCP cloud.")
    parser.add_argument(
        "--checkpoints-dir",
        help="Custom bucket to use to store model checkpoints created by Ray Tune.")
    parser.add_argument(
        "--code-path",
        help="Path to the code working dir zip package.")
    parser.add_argument(
        "--aws-role-arn",
        help="An AWS IAM ARN for a role that can read/write to your S3 bucket. Only needed for AWS cloud types.")
    args = parser.parse_args()

    assert args.cloud_name and args.checkpoints_dir and args.code_path, (
        "Please specify all of --cloud-name, --checkpoints-dir, and --code-path")

    sdk = AnyscaleSDK()
    epoch = datetime.now().strftime('%Y%m%d-%H%M%S')

    cluster_env, cluster_compute = build_cfgs(args, epoch)

    submit_train_job(args, cluster_env, cluster_compute, epoch)
    serve_job_id = submit_deploy_job(args, cluster_env, cluster_compute, epoch)

    serve_job = sdk.get_production_job(production_job_id=serve_job_id).result
    cluster_id = serve_job.state.cluster_id

    p = parse.urlparse(
        sdk.get_session(cluster_id).result.ray_dashboard_url)
    token = parse.parse_qs(p.query)['token'][0]
    p = p._replace(path="/serve/corridor", query="")
    serve_url = parse.urlunparse(p)

    print("Done")
    print("Your serve job id is: ", serve_job_id)
    print("Your deployment is at: ", serve_url)
    print("Your access token is: ", token)

    test_deployment(serve_url, token)

    # Clean up deployment after everything is done.
    sdk.terminate_job(production_job_id=serve_job_id)
