#!/bin/bash


if [ -n "$AG_BENCH_DEV_URL" ]; then
    echo "Using Development Branch: $AG_BENCH_DEV_URL" >&2
    agbench run $config_file --dev-branch $AG_BENCH_DEV_URL
else
    echo "Using Released autogluon.bench: " >&2
    agbench run $config_file
fi
