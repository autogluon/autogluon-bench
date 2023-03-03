from abc import ABC, abstractmethod
import json
import os
import time

class Benchmark(ABC):
    def __init__(self, module: str, benchmark_name: str, root_dir: str):
        current_time = time.localtime()
        formatted_time = time.strftime("%Y%m%dT%H%M%S", current_time)
        self.module = module
        self.benchmark_name = benchmark_name + "_" + formatted_time
        self.benchmark_dir = os.path.join(root_dir, self.benchmark_name)
        self.metrics_path = os.path.join(self.benchmark_dir, "results")


    @abstractmethod
    def run(self):
        pass


    @abstractmethod
    def save_metrics(self, metrics):
        pass


    def upload_metrics(self, s3_bucket, s3_directory):
        import boto3
        s3 = boto3.client('s3')
        for root, dirs, files in os.walk(self.metrics_path):
            for filename in files:
                local_path = os.path.join(root, filename)

                relative_path = os.path.relpath(local_path, self.metrics_path)
                s3_path = os.path.join(s3_directory, relative_path)

                # upload the file to S3
                s3.upload_file(local_path, s3_bucket, s3_path)
            