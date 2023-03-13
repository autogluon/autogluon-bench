import argparse
import subprocess


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--git_uri', type=str, help='GitHub URI')
    parser.add_argument('--git_branch', type=str, default='master', help='Git branch to checkout (default: main)')
    parser.add_argument('--module', type=str, choices=['tabular', 'multimodal', 'ts'], help='AutoGluon modules.')
    parser.add_argument('--mode', type=str, choices=['local', 'aws'], help='Where to run benchmark.')
    parser.add_argument('--data_path', type=str, help='Can be one of: dataset name, local path, S3 path, AMLB task ID/name')
    parser.add_argument('--benchmark_name', type=str, help='A unique name for the benchmark run.')
    parser.add_argument('--framework', choices=["AutoGluon", "AutoGluon_bestquality", "AutoGluon_hq", "AutoGluon_gq"], type=str, help='AMLB framework to evaluate')
    parser.add_argument('--label', type=str, choices=['stable', 'latest'], help='Labeled version of framework.')
    parser.add_argument('--amlb_benchmark', type=str, help='AMLB benchmark type.')
    parser.add_argument('--amlb_constraint', type=str, help='AMLB resource constraints.')
    parser.add_argument('--amlb_task', type=str, help='Task name. When OpenML is used, dataset name should be used.')

    args = parser.parse_args()
    return args
    
def construct_cmd(args):
    # Define a dictionary mapping each optional argument to its corresponding named argument in the shell command
    arg_map = {
        'git_uri': '--git_uri', 
        'git_branch': '--git_branch', 
        'module': '--module', 
        'data_path': '--data_path',
        'benchmark_name': '--benchmark_name',
        'framework': '--framework',
        'label': '--label',
        'amlb_benchmark': '--amlb_benchmark',
        'amlb_constraint': '--amlb_constraint',
        'amlb_task': '--amlb_task',
    }

    # Construct the shell command with the parsed arguments
    command = []
    filtered_opts = ['mode']
    for arg, value in vars(args).items():
        if arg not in filtered_opts and value is not None:
            command.append(arg_map[arg])
            command.append(str(value))

    # Run the shell command using subprocess
    return command


if __name__ == '__main__':
    args = get_args()
    print(args)
    
    if args.mode == "aws":
        raise NotImplementedError
        # deploy_stack()
        # TODO: invoke lambda to start the container jobs on AWS Batch
        # destroy_stack()
    elif args.mode == "local":
        command = ["./run_benchmarks.sh"] + construct_cmd(args)            
        subprocess.run(command)
    else:
        raise NotImplementedError
