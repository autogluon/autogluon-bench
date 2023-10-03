Contents of each metadata file:

### task_metadata.csv

Contains the AutoMLBenchmark 104 datasets (study 271 and 269)

Generated from `autogluon_benchmark.metadata.metadata_generator.generate_metadata(study=[271, 269])`

### task_metadata_244.csv

Filtered version of `task_metadata_289.csv`.

This version is preferred over `289`, as it has removed corrupted datasets and trivially easy datasets.

The final result is 244 datasets which are not corrupted and are not trivial.

Tasks are in the `10-fold Crossvalidation` estimation_procedure.

You can regenerate this file by running `scripts/gen_task_metadata/run_gen_task_metadata_244.py`.

### task_metadata_289.csv

#### NOTE: Please use task_metadata_244.csv, which is the cleaned version of this metadata.

Contains the AutoMLBenchmark 104 datasets (study 271 and 269), plus the 208 datasets from study 293.

These 312 datasets were then deduped by `did` resulting in 279 datasets.

Then, all tasks were standardized to `10-fold Crossvalidation` estimation_procedure (`estimation_procedure_id=1`)

Note that these 279 datasets might need further cleaning, as some datasets from study 293 appear to be corrupted.

(Named 289 because originally there were 289 datasets, but have since been further deduped to 279)

You can regenerate this file by running `scripts/gen_task_metadata/run_gen_task_metadata_289.py`.

### task_metadata_old.csv

Contains the AutoMLBenchmark 104 datasets (study 271 and 269) prior to dataset ID fixes/updates

4 datasets have a different `tid` than they do in `task_metadata.csv`, but are the same tasks in practice.

This is needed to fix any `tid` mismatches from old runs of AMLB before the changes.

Differences occur in the following 4 datasets with `tid`:

```
Name                | oldtid | newtid

KDDCup09-Upselling  | 360115 | 360975
MIP-2016-regression | 359947 | 360945
QSAR-TID-10980      |  14097 | 360933
QSAR-TID-11         |  13854 | 360932
```
