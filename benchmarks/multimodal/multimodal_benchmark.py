from ..benchmark import Benchmark
from typing import List
from datetime import datetime
import subprocess
import os
import pandas as pd
import torchvision.datasets as data
import json


class MultiModalBenchmark(Benchmark):
    def __init__(
        self, 
        benchmark_name: str, 
        root_dir: str = "./benchmarks/multimodal/benchmark_runs",
    ):
        super().__init__(
            module="multimodal",
            benchmark_name=benchmark_name,
            root_dir=root_dir,
        )

    def setup(
        self, 
        git_user: str = "autogluon", 
        git_branch: str = "master",
    ):
        command = ["./autogluon_bench/multimodal/setup.sh", git_user, git_branch, self.benchmark_dir]
        subprocess.run(command, stdout=subprocess.PIPE)


    def _convert_torchvision_dataset(self, dataset):
        import pandas as pd
        from io import BytesIO

        data = []
        for i in range(len(dataset)):
            x, y = dataset[i]
            img_byte_arr = BytesIO()
            x.save(img_byte_arr, format="PNG")
            img_byte_arr = img_byte_arr.getvalue()
            data.append((img_byte_arr, y))
        df = pd.DataFrame(data, columns=['image', 'label'])
        return df
    
    
    def _load_dataset(
        self,
        data_path: str,
    ):
        # TODO: can also accept custom dataframes that passed in
        
        if os.path.isdir(data_path):
            raise NotImplementedError
        elif data_path.startswith('s3://'):
            raise NotImplementedError
        else:
            if data_path == "MNIST":
                train_data = data.MNIST("./data", train=True, download=True)
                test_data = data.MNIST("./data", train=False, download=True)
                train_data = self._convert_torchvision_dataset(train_data)
                test_data = self._convert_torchvision_dataset(test_data)
                return train_data, test_data
            else:
                raise NotImplementedError
    

    def save_metrics(self, metrics):
        file = os.path.join(self.metrics_path, "metrics.json")
        with open(file, "w") as f:
            json.dump(metrics, f, indent=2)
            print(f"metrics saved to {self.metrics_path}")
        f.close()


    def run(
        self,
        train_data: pd.DataFrame = None,  # custom dataset
        test_data: pd.DataFrame = None,
        data_path: str = None,
        problem_type: str = None,
        label: str = "label",
        presets: str = "best_quality",
        metrics: List[str] = ["acc"],
        time_limit: int = 10,
        hyperparameters: dict = None,
        # TODO: replace with config yaml
    ):
        from autogluon.multimodal import MultiModalPredictor
        
        if data_path:
            train_data, test_data = self._load_dataset(data_path=data_path)

        predictor = MultiModalPredictor(
            label=label,
            problem_type=problem_type,
            presets=presets,
        )
        predictor.fit(
            train_data=train_data,
            hyperparameters=hyperparameters,
            time_limit=time_limit,
        )
        scores = predictor.evaluate(test_data)
        timestamp = datetime.now()
        metrics = {
            "benchmark_name": self.benchmark_name,
            "problem_type": predictor.problem_type,
            "scores": scores,
            "timestamp": timestamp.strftime("%H:%M:%S")
        }
        self.metrics = metrics
        return self.metrics