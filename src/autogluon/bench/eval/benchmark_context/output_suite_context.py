from __future__ import annotations

import copy
import os
from typing import List, Optional, Set

import pandas as pd
import ray
from tqdm import tqdm

from autogluon.bench.eval.benchmark_context.output_context import OutputContext
from autogluon.bench.eval.benchmark_context.utils import get_s3_paths
from autogluon.common.loaders import load_pd

DEFAULT_COLUMNS_TO_KEEP = [
    "id",
    "task",
    "framework",
    "constraint",
    "fold",
    "type",
    "metric",
    "mode",
    "version",
    "params",
    "app_version",
    "utc",
    "seed",
]


class OutputSuiteContext:
    def __init__(
        self,
        path: str,
        paths_extra: List[str] = None,
        contains: Optional[str] = None,
        allowed_tids: Optional[Set[int]] = None,
        columns_to_keep: Optional[List[str]] = None,
        include_infer_speed: bool = False,
        keep_params: bool = True,
        mode: str = "seq",
        max_workers: int = 64,
    ):
        """
        Parameters
        ----------
        path : str
            The S3 path to the output folder of an AMLB run
            Example: "s3://automl-benchmark-ag/ec2/2023_02_27_zs/"
        paths_extra: List[str], default = None
            A list of additional paths to use when generating `self.output_contexts`.
            Additional output contexts will be identified in the same way as done with `path`.
        contains : Optional[str], default = None
            Can be specified to limit the returned outputs.
            For example, by specifying the constraint, such as ".1h8c."
        allowed_tids : Optional[Set[int]], default = None
            If specified, results will be filtered to only OutputContext objects
            that have an OpenML TID within `allowed_tids`.
        columns_to_keep : List[str], default = None
            The list of columns to keep when loading from results files.
            If None, uses DEFAULT_COLUMNS_TO_KEEP
        include_infer_speed : bool, default = False
            Whether to merge infer_speed results when loading results and leaderboard outputs
        keep_params : bool, default = True
            If False, drops the "params" column from leaderboard output to reduce size of artifact
        mode : str, default = 'seq'
            One of ['seq', 'ray']
            If 'ray' is installed, this will be much faster as data loading will be parallelized.
            'seq' (sequential) should work on all systems, but will be slower than 'ray'.
            'seq' will work properly in a debugger, whereas 'ray' is difficult to debug.
        max_workers : int, default = 64
            If `mode` is set to 'ray', this restricts the maximum number of ray workers to this value.
            No effect if the number of logical cores is less than this value.
            Values greater than 64 can lead to strange errors likely due to S3 service throttling.
        """
        self._path = path
        self.contains = contains
        self.allowed_tids = allowed_tids
        if self.allowed_tids is not None:
            assert isinstance(self.allowed_tids, set)
            for tid in self.allowed_tids:
                assert isinstance(tid, int)
        self.output_contexts = self.get_output_contexts(contains=self.contains, paths_extra=paths_extra)
        if len(self.output_contexts) == 0:
            print(f"WARNING: No output contexts found via path={self._path}, contains={self.contains}")
        self.include_infer_speed = include_infer_speed
        self.mode = mode
        self.max_workers = max_workers
        if columns_to_keep is None:
            columns_to_keep = DEFAULT_COLUMNS_TO_KEEP
        columns_to_keep = copy.deepcopy(columns_to_keep)
        self.keep_params = keep_params
        if not keep_params:
            columns_to_keep = [c for c in columns_to_keep if c != "params"]
        self.columns_to_keep = columns_to_keep

    def get_output_contexts(self, contains: str = None, paths_extra: List[str] = None) -> List[OutputContext]:
        """
        Parameters
        ----------
        contains : str, default = None
            Can be specified to limit the returned outputs.
            For example, by specifying the constraint, such as ".1h8c."
        paths_extra : List[str], default = None
            An optional list of paths to get additional contexts.
        """
        output_contexts = self._get_output_contexts(path=self.path, contains=contains)
        if paths_extra is not None:
            for path in paths_extra:
                output_contexts += self._get_output_contexts(path=path, contains=contains)
        return output_contexts

    @staticmethod
    def _get_output_contexts(path: str, contains: str = None) -> List[OutputContext]:
        """
        Parameters
        ----------
        contains : str, default = None
            Can be specified to limit the returned outputs.
            For example, by specifying the constraint, such as ".1h8c."
        """
        paths_to_results = get_s3_paths(path, contains=contains, suffix="scores/results.csv")
        output_contexts = [OutputContext.from_results_path(path=results_path) for results_path in paths_to_results]
        return output_contexts

    @property
    def path(self):
        return self._path

    @property
    def num_contexts(self):
        return len(self.output_contexts)

    def _loop_func(
        self, func, input_list: list, kwargs=None, allow_exception=False, exception_default=None, mode=None
    ) -> list:
        if len(input_list) == 0:
            return []
        if mode is None:
            mode = self.mode
        process_func = _with_ray if mode == "ray" else _with_seq
        return process_func(
            func=func,
            input_list=input_list,
            kwargs=kwargs,
            allow_exception=allow_exception,
            exception_default=exception_default,
            num_cpus=self.max_workers,
        )

    def load_results(self) -> List[pd.DataFrame]:
        return self._loop_func(
            func=OutputContext.load_results,
            input_list=self.output_contexts,
            kwargs=dict(
                include_infer_speed=self.include_infer_speed,
                keep_params=self.keep_params,
                allowed_tids=self.allowed_tids,
            ),
        )

    def load_zeroshot_metadata(self, max_size_mb: float = None, allow_exception=False) -> List[dict]:
        return self._loop_func(
            func=OutputContext.load_zeroshot_metadata,
            input_list=self.output_contexts,
            kwargs=dict(max_size_mb=max_size_mb),
            allow_exception=allow_exception,
        )

    def get_zeroshot_metadata_size_bytes(self, allow_exception=False) -> List[int]:
        return self._loop_func(
            func=OutputContext.get_zeroshot_metadata_size_bytes,
            input_list=self.output_contexts,
            allow_exception=allow_exception,
        )

    def filter_failures(self):
        amlb_info_list = self.get_amlb_info()
        output_contexts_valid = []
        for info, output_context in zip(amlb_info_list, self.output_contexts):
            if info is None:
                output_contexts_valid.append(output_context)
        print(f"Filtered Failures: {len(output_contexts_valid)}/{len(self.output_contexts)} valid")
        self.output_contexts = output_contexts_valid

    def filter(self, filter_lst: List[bool], inplace=True):
        """
        Filter self.output_contexts by a boolean list. Only keep contexts where the boolean is True.
        """
        if not inplace:
            out = copy.deepcopy(self)
        else:
            out = self

        assert len(filter_lst) == len(out.output_contexts)
        out.output_contexts = [
            output_context for output_context, is_valid in zip(out.output_contexts, filter_lst) if is_valid is True
        ]

        return out

    def aggregate_results(self, results_list: List[pd.DataFrame] | None = None) -> pd.DataFrame:
        if results_list is None:
            results_list = self.load_results()
        results_df = pd.concat(results_list, ignore_index=True)
        return results_df

    def load_leaderboards(self) -> List[pd.DataFrame]:
        if self.num_contexts == 0:
            raise AssertionError("Empty output_contexts!")

        kwargs = dict(
            output_contexts=self.output_contexts,
            columns_to_keep=self.columns_to_keep,
            with_infer_speed=self.include_infer_speed,
        )

        # TODO: Migrate to `self._loop_func`
        if self.mode == "seq":
            result = self._aggregate_leaderboards_seq(**kwargs)
        elif self.mode == "ray":
            result = self._aggregate_leaderboards_ray(**kwargs)
        else:
            raise ValueError(f'Unsupported mode "{self.mode}"')
        print(
            f"Successfully loaded {len(result)}/{self.num_contexts} task outputs "
            f"({round(100 * (len(result) / self.num_contexts), 1)}%)..."
        )
        return result

    def aggregate_leaderboards(self, leaderboards_list: List[pd.DataFrame] | None = None) -> pd.DataFrame:
        if leaderboards_list is None:
            leaderboards_list = self.load_leaderboards()
        leaderboards_df = pd.concat(leaderboards_list, ignore_index=True)
        return leaderboards_df

    def load_model_failures(self) -> List[pd.DataFrame]:
        return self._loop_func(
            func=OutputContext.get_model_failures, input_list=self.output_contexts, allow_exception=True
        )

    def load_info_file_sizes(self, sum: bool = False) -> List[pd.DataFrame]:
        return self._loop_func(
            func=OutputContext.get_info_file_sizes,
            input_list=self.output_contexts,
            kwargs={"sum": sum},
            allow_exception=True,
        )

    def aggregate_info_file_sizes(self, sum: bool = False) -> pd.DataFrame:
        info_file_sizes_lst = self.load_info_file_sizes(sum=sum)
        none_count = 0
        for e in info_file_sizes_lst:
            if e is None:
                none_count += 1
        if none_count == len(info_file_sizes_lst):
            return pd.DataFrame()
        else:
            info_file_sizes_df = pd.concat(info_file_sizes_lst, ignore_index=True)
            return info_file_sizes_df

    def aggregate_model_failures(self) -> pd.DataFrame:
        model_failures_list = self.load_model_failures()
        none_count = 0
        for e in model_failures_list:
            if e is None:
                none_count += 1
        if none_count == len(model_failures_list):
            return pd.DataFrame()
        else:
            model_failures_df = pd.concat(model_failures_list, ignore_index=True)
            return model_failures_df

    def get_logs(self) -> List[str]:
        return self._loop_func(func=OutputContext.get_logs, input_list=self.output_contexts)

    def aggregate_logs(self) -> str:
        log_list = self.get_logs()
        return "\n\n".join(log_list)

    def get_amlb_info(self) -> List[str]:
        return self._loop_func(func=OutputContext.get_amlb_info, input_list=self.output_contexts)

    def get_benchmark_failures(self):
        amlb_info_dict = dict()

        amlb_info_list = self.get_amlb_info()

        for output_context, amlb_info in zip(self.output_contexts, amlb_info_list):
            path_relative = remove_prefix(output_context.path, prefix=self.path)

            if amlb_info is not None:
                if amlb_info in amlb_info_dict:
                    amlb_info_dict[amlb_info].append(path_relative)
                else:
                    amlb_info_dict[amlb_info] = [path_relative]
        amlb_info_count_dict = dict()
        for info, paths in amlb_info_dict.items():
            amlb_info_count_dict[info] = len(paths)
        sorted_info = sorted(amlb_info_count_dict.items(), key=lambda x: x[1], reverse=True)
        sorted_info = [i[0] for i in sorted_info]
        for i in sorted_info:
            count = amlb_info_count_dict[i]
            print(f"{count} | {i}\n" f"\t{amlb_info_dict[i]}")

    @staticmethod
    def _aggregate_leaderboards_ray(output_contexts, columns_to_keep, with_infer_speed):
        num_contexts = len(output_contexts)
        if not ray.is_initialized():
            ray.init()
        results = []
        for i, output_context in enumerate(output_contexts):
            results.append(
                get_single_leaderboard_ray.remote(output_context, columns_to_keep, with_infer_speed, i, num_contexts)
            )
        result = ray.get(results)
        result = [r for r in result if r is not None]
        return result

    @staticmethod
    def _aggregate_leaderboards_seq(output_contexts, columns_to_keep, with_infer_speed):
        num_contexts = len(output_contexts)
        results = []
        for i, output_context in enumerate(output_contexts):
            results.append(
                get_single_leaderboard_seq(output_context, columns_to_keep, with_infer_speed, i, num_contexts)
            )
        results = [r for r in results if r is not None]
        return results

    def construct_zs_dict(self, results_lst=None, zeroshot_metadata_list=None):
        if results_lst is None:
            results_lst = self.load_results()
        if zeroshot_metadata_list is None:
            zeroshot_metadata_list = self.load_zeroshot_metadata()
        return self._construct_zs_dict(results_lst=results_lst, zeroshot_metadata_list=zeroshot_metadata_list)

    @classmethod
    def _construct_zs_dict(cls, results_lst, zeroshot_metadata_list):
        num_paths = len(results_lst)
        aggregated_pred_proba = {}
        aggregated_ground_truth = {}
        for i, (zeroshot_metadata, scores) in enumerate(zip(zeroshot_metadata_list, results_lst)):
            id = scores["id"][0]
            fold = scores.iloc[0]["fold"]
            task_name = scores.iloc[0]["task"]
            fold = int(fold)
            print(f"{i + 1}/{num_paths} | {task_name} | {fold}")

            if task_name not in aggregated_ground_truth:
                aggregated_ground_truth[task_name] = {}
            if fold not in aggregated_ground_truth[task_name]:
                aggregated_ground_truth[task_name][fold] = {}
                for k in [
                    "y_val",
                    "y_test",
                    "eval_metric",
                    "problem_type",
                    "ordered_class_labels",
                    "ordered_class_labels_transformed",
                    "problem_type_transform",
                    "num_classes",
                    "label",
                ]:
                    aggregated_ground_truth[task_name][fold][k] = zeroshot_metadata[k]
                aggregated_ground_truth[task_name][fold]["task"] = task_name
                aggregated_ground_truth[task_name][fold]["id"] = id
            if task_name not in aggregated_pred_proba:
                aggregated_pred_proba[task_name] = {}
            if fold not in aggregated_pred_proba[task_name]:
                aggregated_pred_proba[task_name][fold] = {}
            for k in ["pred_proba_dict_val", "pred_proba_dict_test"]:
                if k not in aggregated_pred_proba[task_name][fold]:
                    aggregated_pred_proba[task_name][fold][k] = {}
                for m, pred_proba in zeroshot_metadata[k].items():
                    if aggregated_ground_truth[task_name][fold]["problem_type"] == "binary":
                        if isinstance(pred_proba, pd.DataFrame):
                            assert len(pred_proba.columns) == 2
                            pred_proba = pred_proba[1]
                        assert isinstance(pred_proba, pd.Series)
                    aggregated_pred_proba[task_name][fold][k][m] = pred_proba
        return aggregated_pred_proba, aggregated_ground_truth


def _with_seq(
    func, input_list: list, kwargs=None, allow_exception=False, exception_default=None, num_cpus=None
) -> list:
    """
    For-loop through a function call sequentially
    """
    if kwargs is None:
        kwargs = dict()
    if allow_exception:

        def _func(*args, **kw):
            try:
                return func(*args, **kw)
            except:
                return exception_default

    else:
        _func = func
    out_list = []
    for input_val in input_list:
        out_list.append(_func(input_val, **kwargs))
    return out_list


def _with_ray(
    func,
    input_list: list,
    kwargs=None,
    allow_exception=False,
    exception_default=None,
    num_cpus: int = None,
    progress_bar: bool = True,
) -> list:
    """
    For-loop through a function call with ray
    """
    if kwargs is None:
        kwargs = dict()
    if allow_exception:

        def _func(*args, **kw):
            try:
                return func(*args, **kw)
            except:
                return exception_default

    else:
        _func = func

    if not ray.is_initialized():
        num_cpus_system = os.cpu_count()
        if num_cpus is not None and num_cpus_system > num_cpus:
            # Restrict to avoid strange errors, probably S3 throttling
            ray.init(num_cpus=num_cpus)
        else:
            ray.init()
    remote_func = ray.remote(_func)
    results = []
    for i in input_list:
        results.append(remote_func.remote(i, **kwargs))
    if progress_bar:
        result = [ray.get(res) for res in tqdm(results)]
    else:
        result = ray.get(results)
    return result


@ray.remote
def get_single_leaderboard_ray(output_context: OutputContext, columns_to_keep, with_infer_speed, i, num_contexts):
    return output_context.get_single_leaderboard(columns_to_keep, with_infer_speed, i, num_contexts=num_contexts)


def get_single_leaderboard_seq(output_context: OutputContext, columns_to_keep, with_infer_speed, i, num_contexts):
    return output_context.get_single_leaderboard(columns_to_keep, with_infer_speed, i, num_contexts=num_contexts)


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix) :]
    else:
        raise AssertionError("Lacking prefix!")
