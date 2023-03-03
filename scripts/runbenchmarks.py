import argparse
import subprocess
from ..stacks.run_deploy import deploy_stack

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--action', type=str, choices=['run', 'list', 'cancel'], help='Actions.')
    parser.add_argument('--git_user', type=str, help='GitHub username for cloning the repository')
    parser.add_argument('--git_branch', type=str, default='master', help='Git branch to checkout (default: main)')
    parser.add_argument('--module', type=str, choices=['tabular', 'multimodal', 'ts'], help='AutoGluon modules.')
    parser.add_argument('--mode', type=str, choices=['local', 'aws'], help='Where to run benchmark.')
    parser.add_argument('--data_path', type=str, help='Can be one of: dataset name, local path, S3 path, AMLB task ID/name')
    parser.add_argument('--benchmark_name', type=str, help='A unique name for the benchmark run.')


    args = parser.parse_args()
    return args
    

if __name__ == '__main__':
    args = get_args()
    
    if args.action != "run":
        raise NotImplementedError
    
    if args.mode == "aws":
        deploy_stack()
        # TODO: invoke lambda to start the container jobs on AWS Batch
        # destroy_stack()
    elif args.mode == "local":
        command = ["./benchmarks/run_benchmarks.sh", args.git_user, args.git_branch, args.module, args.data_path, args.benchmark_name]
        subprocess.run(command, stdout=subprocess.PIPE)
