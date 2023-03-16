import os
import time
from abc import ABC, abstractmethod


class Benchmark(ABC):
    def __init__(self, benchmark_name: str, root_dir: str):
        current_time = time.localtime()
        formatted_time = time.strftime("%Y%m%dT%H%M%S", current_time)
        self.benchmark_name = benchmark_name + "_" + formatted_time
        self.benchmark_dir = os.path.join(root_dir, self.benchmark_name)
        self.metrics_dir = os.path.join(self.benchmark_dir, "results")

    @abstractmethod
    def run(self):
        pass

    def upload_metrics(self, s3_bucket: str, s3_dir: str):
        import boto3
        
        print(f'Saving metrics to S3 Bucket {s3_bucket}...')
        s3 = boto3.client("s3")
        for root, dirs, files in os.walk(self.metrics_dir):
            for filename in files:
                local_path = os.path.join(root, filename)

                relative_path = os.path.relpath(local_path, self.metrics_dir)
                s3_path = os.path.join(s3_dir, relative_path)

                # upload the file to S3
                s3.upload_file(local_path, s3_bucket, s3_path)
        
        print(f"Metrics under {self.metrics_dir} has been saved to {s3_bucket}/{s3_dir}.")
