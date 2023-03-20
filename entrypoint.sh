#!/bin/bash

python runbenchmarks.py \
--git_uri $git_uri \
--git_branch $git_branch \
--module $module \
--mode $mode \
--data_path $data_path \
--benchmark_name $benchmark_name \
--framework $framework \
--label $label \
--amlb_benchmark $amlb_benchmark \
--amlb_constraint $amlb_constraint \
--amlb_task $amlb_task \
--s3_bucket $s3_bucket \
--config_file $config_file \