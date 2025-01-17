import argparse
import os
import uuid
from pathlib import Path

import main as classification
import submitit


def parse_args():
    classification_parser = classification.get_args_parser()
    parser = argparse.ArgumentParser("Submitit for ConvNeXt", parents=[classification_parser])
    parser.add_argument("--ngpus", default=8, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--gpu", default=None, type=str, help="gpu name")
    parser.add_argument("--nodes", default=2, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=72, type=int, help="Duration of the job, in hours")
    parser.add_argument("--job_name", default="convnext", type=str, help="Job name")
    parser.add_argument("--job_dir", default="", type=str, help="Job directory; leave empty for default")
    parser.add_argument("--partition", default=None, type=str, help="Partition where to submit")
    parser.add_argument("--reservation", default=None, type=str, help="Partition where to submit")
    parser.add_argument("--account", default=None, type=str, help="account where to submit")
    parser.add_argument("--use_volta32", action='store_true', default=False, help="Big models? Use this")
    parser.add_argument('--comment', default="", type=str,
                        help='Comment to pass to scheduler, e.g. priority message')
    return parser.parse_args()


def get_shared_folder() -> Path:
    if Path("/home/mlr/projects/def-pal/mlr/ConvNeXt").is_dir():
        p = Path(f"/home/mlr/projects/def-pal/mlr/ConvNeXt/checkpoint")
        p.mkdir(exist_ok=True)
        return p
    elif Path("/home/mila/m/mats-leon.richter/scratch/ConvNeXt").is_dir():
        p = Path("/home/mila/m/mats-leon.richter/scratch/ConvNeXt/checkpoint")
        p.mkdir(exist_ok=True)
        return p
    elif Path("/scratch/mlrichter/").is_dir():
        p = Path("/scratch/mlrichter/checkpoint")
        p.mkdir(exist_ok=True)
        return p

    raise RuntimeError("No shared folder available")


def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import main as classification

        self._setup_gpu_args()
        classification.main(self.args)

    def checkpoint(self):
        import os
        import submitit

        self.args.dist_url = get_init_file().as_uri()
        self.args.auto_resume = True
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(self.args.job_dir)
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main():
    args = parse_args()

    if args.job_dir == "":
        args.job_dir = get_shared_folder() / "%j"

    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout * 60

    partition = args.partition
    kwargs = {}
    if args.use_volta32:
        kwargs['slurm_constraint'] = 'v100l'
    elif args.gpu:
        kwargs['slurm_constraint'] = args.gpu
    if args.comment:
        kwargs['slurm_comment'] = args.comment
    if args.account:
        kwargs['slurm_account'] = args.account
    if args.reservation is not None:
        kwargs['reservation'] = 'DGXA100'

    executor.update_parameters(
        mem_gb=64 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=6,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        **kwargs
    )

    executor.update_parameters(name=args.job_name)

    args.dist_url = get_init_file().as_uri()
    args.output_dir = args.job_dir

    trainer = Trainer(args)
    job = executor.submit(trainer)

    print("Submitted job_id:", job.job_id)


if __name__ == "__main__":
    main()
