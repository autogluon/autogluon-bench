from typing import List, Optional, Set

import boto3
import numpy as np
import pandas as pd

from autogluon.common.loaders import load_pd, load_pkl
from autogluon.common.utils.s3_utils import s3_path_to_bucket_prefix


class OutputContext:
    def __init__(self, path):
        """
        Parameters
        ----------
        path : str
            The S3 path to the output folder of an AMLB task
        """
        self._path = path

    @classmethod
    def from_results_path(cls, path):
        results_suffix = "/scores/results.csv"
        assert path.endswith(results_suffix)
        dataset_directory = path.rsplit(results_suffix, 2)[0] + "/"
        return cls(path=dataset_directory)

    @property
    def path(self):
        return self._path

    def get_s3_bucket(self, path=None):
        if path is None:
            path = self.path
        return s3_path_to_bucket_prefix(path)[0]

    def get_s3_prefix(self, path=None):
        if path is None:
            path = self.path
        return s3_path_to_bucket_prefix(path)[1]

    @property
    def path_results(self):
        return self.path + "scores/results.csv"

    @property
    def path_leaderboard(self):
        return self.path + "leaderboard.csv"

    @property
    def path_infer_speed(self):
        return self.path + "infer_speed.csv"

    @property
    def path_logs(self):
        return self.path + "logs.zip"

    @property
    def path_info(self):
        return self.path + "info/info.pkl"

    @property
    def path_info_file_sizes(self):
        return self.path + "info/file_sizes.csv"

    @property
    def path_zeroshot_metadata(self):
        return self.path + "zeroshot/zeroshot_metadata.pkl"

    def load_results(
        self,
        include_infer_speed: bool = False,
        keep_params: bool = True,
        allowed_tids: Optional[Set[int]] = None,
    ) -> Optional[pd.DataFrame]:
        """
        allowed_tids can be specified to ensure results are only returned if they belong to a specific set of tasks.
        """
        results = load_pd.load(self.path_results)
        if allowed_tids is not None:
            tid = self.get_tid(results_df=results)
            if tid not in allowed_tids:
                return None
        if not keep_params:
            results = results.drop(columns=["params"])
        if include_infer_speed:
            try:
                results = self._merge_results_with_infer_speed(results=results, infer_speed=None)
            except FileNotFoundError:
                return None
        return results

    def load_leaderboard(self) -> pd.DataFrame:
        return load_pd.load(self.path_leaderboard)

    def load_infer_speed(self) -> pd.DataFrame:
        return load_pd.load(self.path_infer_speed)

    def load_zeroshot_metadata(self, max_size_mb: float = None) -> dict:
        s3_bucket = self.get_s3_bucket()
        s3_prefix = self.get_s3_prefix(path=self.path_zeroshot_metadata)
        s3 = boto3.client("s3")
        response = s3.head_object(Bucket=s3_bucket, Key=s3_prefix)
        size = response["ContentLength"]

        size_og_mb = round(size / 1e6, 3)

        if max_size_mb is not None and size_og_mb > max_size_mb:
            print(f"exit: {size_og_mb} | {self.path}")
            return None

        a = load_pkl.load(self.path_zeroshot_metadata)
        print(f"CUR Size: {size_og_mb} MB | {self.path}")
        return a

    def get_tid(self, results_df: pd.DataFrame = None) -> int:
        """
        Get OpenML task ID (tid)
        """
        if results_df is None:
            results_df = self.load_results()
        tid = int(results_df["id"].iloc[0].rsplit("/", 1)[-1])
        return tid

    def get_amlb_info(self, results_df: pd.DataFrame = None) -> str:
        if results_df is None:
            results_df = self.load_results()
        amlb_info = results_df["info"].iloc[0]
        if not isinstance(amlb_info, str) and np.isnan(amlb_info):
            amlb_info = None
        return amlb_info

    def get_single_leaderboard(self, columns_to_keep, with_infer_speed, i, num_contexts):
        print_msg = f"{i + 1}/{num_contexts} | {self.path}"
        scores = self.load_results()
        amlb_info: str = self.get_amlb_info(results_df=scores)
        if amlb_info is not None:
            print_msg = f"{print_msg}\n" f"\tAMLB_INFO: {amlb_info}"
        try:
            leaderboard = self.load_leaderboard()
            leaderboard = leaderboard.drop(columns=["features"], errors="ignore")
            if with_infer_speed:
                leaderboard = self._merge_leaderboard_with_infer_speed(leaderboard=leaderboard)
        except Exception as e:
            print(f"FAILURE: {print_msg}\n" f"\t{e.__class__.__name__}: {e}")
            return None
        else:
            result_val = scores.iloc[0]["result"]
            if np.isnan(result_val):
                print(
                    f"FAILURE (STRANGE!): {print_msg}\n"
                    f"\tPRIORITIZE DEBUGGING: This is a strange error!\n"
                    f"\t\tDespite leaderboard.csv existing, the overall result is NaN, indicating a failure. "
                    f"This is likely a bug in the dataset metadata, such as columns being misaligned, "
                    f"and not a bug with AutoGluon specifically."
                )
                return None

            # scores = scores[scores['fold'] == 0]
            # print(scores)
            scores = scores[columns_to_keep]
            scores = scores.rename(columns={"framework": "framework_parent"})

            # best_compressed = leaderboard[leaderboard['model'].str.contains('_FULL')]
            # best_distilled = leaderboard[leaderboard['model'].str.contains('_d1')].sort_values('score_val', ascending=False).head(1)
            best_weighted = (
                leaderboard[leaderboard["model"].str.contains("WeightedEnsemble_")]
                .sort_values("score_val", ascending=False)
                .head(1)
            )
            best_nonweighted = (
                leaderboard[~leaderboard["model"].str.contains("WeightedEnsemble_")]
                .sort_values("score_val", ascending=False)
                .head(1)
            )

            # best_compressed['model'] = 'autogluon_compressed'
            # best_distilled['model'] = 'autogluon_distilled'
            # FIXME: Doesn't work for refit_full!!! score_val is NaN!
            best_weighted["model"] = "autogluon_ensemble"
            best_nonweighted["model"] = "autogluon_single"
            # print(best_compressed)
            # print(best_distilled)
            # print(best_weighted)

            combined = pd.concat(
                [
                    leaderboard,
                    best_weighted,
                    # best_compressed,
                    # best_distilled,
                    best_nonweighted,
                ],
                ignore_index=True,
            )
            # combined = combined.sort_values('score_test', ascending=False).reset_index(drop=True)
            combined["id"] = scores["id"][0]
            # print(combined)

            combined_full = pd.merge(combined, scores, on="id", how="left")

            combined_full["framework"] = combined_full["model"]
            # combined_full['result'] = combined_full['score_test']
            combined_full["duration"] = combined_full["fit_time"]
            # combined_full['predict_duration'] = combined_full['pred_time_test']

            # print(combined_full)
            print(f"SUCCESS: {print_msg}")
            return combined_full

    def _merge_leaderboard_with_infer_speed(self, leaderboard: pd.DataFrame) -> pd.DataFrame:
        infer_speed_df = self.load_infer_speed()
        infer_speed_df = infer_speed_df[["model", "batch_size", "pred_time_test_with_transform"]]
        infer_speed_m_df = infer_speed_df.set_index(["model", "batch_size"], drop=True)
        a = infer_speed_m_df.to_dict()
        b = a["pred_time_test_with_transform"]
        c = dict()
        for key_pair, pred_time_test_with_transform in b.items():
            m = key_pair[0]
            bs = key_pair[1]
            col_name = f"pred_time_test_with_transform_{bs}"
            if col_name not in c:
                c[col_name] = {}
            c[col_name][m] = pred_time_test_with_transform
        c_df = pd.DataFrame(c).rename_axis("model").reset_index(drop=False)
        leaderboard = pd.merge(leaderboard, c_df, on="model")
        return leaderboard

    def _merge_results_with_infer_speed(self, results: pd.DataFrame, infer_speed: pd.DataFrame = None) -> pd.DataFrame:
        if infer_speed is None:
            infer_speed = self.load_infer_speed()
        infer_speed_best = infer_speed[infer_speed["model"] == "best"]
        batch_sizes = list(infer_speed_best["batch_size"].unique())
        for batch_size in batch_sizes:
            results[f"pred_time_test_with_transform_{batch_size}"] = infer_speed_best[
                infer_speed_best["batch_size"] == batch_size
            ].iloc[0]["pred_time_test_with_transform"]
        return results
