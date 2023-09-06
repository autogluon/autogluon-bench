ARG AG_BENCH_BASE_IMAGE
FROM $AG_BENCH_BASE_IMAGE

ENV DEBIAN_FRONTEND=noninteractive

# Install essential packages and Python 3.9
RUN apt-get update && \
    apt-get install -y software-properties-common build-essential && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.9 python3.9-dev python3.9-distutils python3.9-venv && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Install utilities and AWS CLI
RUN apt-get install -y python3-pip unzip curl git pciutils && \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip -q awscliv2.zip && \
    ./aws/install && \
    rm awscliv2.zip && \
    python3 -m pip install --upgrade pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /usr/local/aws

# Application-specific steps
ARG AG_BENCH_DEV_URL
ARG AG_BENCH_VERSION
ARG CDK_DEPLOY_REGION
ARG FRAMEWORK_PATH
ARG GIT_URI
ARG GIT_BRANCH
ARG BENCHMARK_DIR
ARG AMLB_FRAMEWORK
ARG AMLB_USER_DIR

WORKDIR /app/

RUN if [ -n "$AG_BENCH_DEV_URL" ]; then \
        echo "Cloning: $AG_BENCH_DEV_URL" \
        && AG_BENCH_DEV_REPO=$(echo "$AG_BENCH_DEV_URL" | cut -d "#" -f 1) \
        && AG_BENCH_DEV_BRANCH=$(echo "$AG_BENCH_DEV_URL" | cut -d "#" -f 2) \
        && git clone --branch "$AG_BENCH_DEV_BRANCH" --single-branch "$AG_BENCH_DEV_REPO" /app/autogluon-bench \
        && python3 -m pip install -e /app/autogluon-bench; \
    else \
        output=$(pip install autogluon.bench==$AG_BENCH_VERSION 2>&1) || true; \
        if echo $output | grep -q "No matching distribution"; then \
            echo -e "ERROR: No matching distribution found for autogluon.bench==$AG_BENCH_VERSION\n\
            To resolve the issue, try 'agbench run <config_file> --dev-branch <autogluon_bench_fork_uri>#<git_branch>'"; \
            exit 1; \
        fi; \
    fi

COPY entrypoint.sh utils/hardware_utilization.sh $FRAMEWORK_PATH/setup.sh custom_configs/ /app/

RUN chmod +x setup.sh entrypoint.sh hardware_utilization.sh \
    && if echo "$FRAMEWORK_PATH" | grep -q -E "tabular|timeseries"; then \
        if [ -n "$AMLB_USER_DIR" ]; then \
            bash setup.sh $GIT_URI $GIT_BRANCH $BENCHMARK_DIR $AMLB_FRAMEWORK $AMLB_USER_DIR; \
        else \
            bash setup.sh $GIT_URI $GIT_BRANCH $BENCHMARK_DIR $AMLB_FRAMEWORK; \
        fi; \
    elif echo "$FRAMEWORK_PATH" | grep -q "multimodal"; then \
        if [ -n "$AG_BENCH_DEV_URL" ]; then \
            bash setup.sh $GIT_URI $GIT_BRANCH $BENCHMARK_DIR --AGBENCH_DEV_URL=$AG_BENCH_DEV_URL; \
        else \
            bash setup.sh $GIT_URI $GIT_BRANCH $BENCHMARK_DIR --AG_BENCH_VER=$AG_BENCH_VERSION; \
        fi; \
    fi \
    && echo "CDK_DEPLOY_REGION=$CDK_DEPLOY_REGION" >> /etc/environment

ENTRYPOINT ["./entrypoint.sh"]
