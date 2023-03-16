import argparse
# from autogluon.bench.cloud.aws.run_deploy import deploy_stack
from autogluon.bench.frameworks.multimodal.multimodal_benchmark import \
    MultiModalBenchmark
from autogluon.bench.frameworks.tabular.tabular_benchmark import \
    TabularBenchmark

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--git_uri', type=str, help='GitHub URI')
    parser.add_argument('--git_branch', type=str, default='master', help='Git branch to checkout (default: main)')
    parser.add_argument('--module', type=str, choices=['tabular', 'multimodal', 'ts'], help='AutoGluon modules.')
    parser.add_argument('--mode', type=str, choices=['local', 'aws'], help='Where to run benchmark.')
    parser.add_argument('--data_path', type=str, help='Can be one of: dataset name, local path, S3 path, AMLB task ID/name')
    parser.add_argument('--benchmark_name', type=str, help='A unique name for the benchmark run.')
    parser.add_argument('--framework', type=str, choices=["AutoGluon", "AutoGluon_bestquality", "AutoGluon_hq", "AutoGluon_gq"], help='AMLB framework to evaluate')
    parser.add_argument('--label', type=str, choices=['stable', 'latest'], help='Labeled version of framework.')
    parser.add_argument('--amlb_benchmark', type=str, help='AMLB benchmark type.')
    parser.add_argument('--amlb_constraint', type=str, help='AMLB resource constraints.')
    parser.add_argument('--amlb_task', type=str, help='Task name. When OpenML is used, dataset name should be used.')
    parser.add_argument('--s3_bucket', type=str, help='S3 bucket to upload metrics and model artifacts.')


    args = parser.parse_args()
    return args
    

def run_benchmark(args):
    if args.module == "multimodal":
        benchmark = MultiModalBenchmark(benchmark_name=args.benchmark_name)
        benchmark.setup(git_uri=args.git_uri, git_branch=args.git_branch)
        benchmark.run(data_path=args.data_path)
        if args.s3_bucket is not None:
            benchmark.upload_metrics(s3_bucket=args.s3_bucket)
        
    elif args.module == "tabular":
        benchmark = TabularBenchmark(
            benchmark_name=args.benchmark_name, 
        )
        benchmark.setup()
        benchmark.run(
            framework=f"{args.framework}:{args.label}",
            benchmark=args.amlb_benchmark,
            constraint=args.amlb_constraint,
            task=args.amlb_task
        )
        if args.s3_bucket is not None:
            benchmark.upload_metrics(s3_bucket=args.s3_bucket, s3_dir=f"{args.module}/{benchmark.benchmark_name}")
    elif args.module == "ts":
        raise NotImplementedError
    

def run():
    args = get_args()
    print(args)
    
    if args.mode == "aws":
        args.mode = "local"
        # deploy_stack()
        # TODO: invoke lambda to start the container jobs on AWS Batch
        # upload metrics to s3
        # destroy_stack()
    elif args.mode == "local":
        run_benchmark(args)
    else:
        raise NotImplementedError
    
 
if __name__ == '__main__':
    run()
