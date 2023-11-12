#!/bin/bash

echo "Running hardware utilization monitoring in the background..."
${AG_BENCH_BASE_DIR}/utils/hardware_utilization.sh &
agbench run $config_file --skip-setup
