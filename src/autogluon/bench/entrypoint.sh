#!/bin/bash

echo "Running hardware utilization monitoring in the background..."
${AGBENCH_BASE}utils/hardware_utilization.sh &

agbench run $config_file --skip-setup
