import argparse

from autogluon.bench.eval.aggregate.results import aggregate_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--s3_bucket', type=str, help="Name of S3 bucket that results to aggregate get outputted to",
                        default='automl-benchmark-ag', nargs='?')
    parser.add_argument('--s3_prefix', type=str, help='Prefix for path to results needing aggregation', default='ec2/', nargs='?')
    parser.add_argument('--version_name', type=str, help='Root folder name in EC2 of results', nargs='?')
    parser.add_argument('--constraint', type=str, help='Name of constraint used in benchmark', default=None, nargs='?')
    parser.add_argument('--include_infer_speed', action='store_true')
    parser.add_argument('--mode', type=str, help='Whether to aggregate via "seq" or via "ray"', default='ray', nargs='?')
    # parser.set_defaults(keep_params=True)
    # parser.set_defaults(include_infer_speed=False)
    # parser.set_defaults(constraint="24h64c")  # FIXME: Remove
    args = parser.parse_args()

    aggregate_results(
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        version_name=args.version_name,
        constraint=args.constraint,
        include_infer_speed=args.include_infer_speed,
        mode=args.mode,
    )
