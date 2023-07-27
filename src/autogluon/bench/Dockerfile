FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.13.1-gpu-py39-cu117-ubuntu20.04-ec2
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
COPY entrypoint.sh utils/gpu_utilization.sh $FRAMEWORK_PATH/setup.sh custom_configs/ /app/

RUN apt-get update && apt-get install -y --no-install-recommends cron unzip curl && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/spool/cron/crontabs \
    && touch /var/log/cron.log \
    && curl "https://d1vvhvl2y92vvt.cloudfront.net/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip -q awscliv2.zip \
    && ./aws/install \
    && rm awscliv2.zip \
    && pip install --upgrade pip

RUN if [ -n "$AG_BENCH_DEV_URL" ]; then \
        echo "Cloning: $AG_BENCH_DEV_URL" \
        && AG_BENCH_DEV_REPO=$(echo "$AG_BENCH_DEV_URL" | cut -d "#" -f 1) \
        && AG_BENCH_DEV_BRANCH=$(echo "$AG_BENCH_DEV_URL" | cut -d "#" -f 2) \
        && git clone --branch "$AG_BENCH_DEV_BRANCH" --single-branch "$AG_BENCH_DEV_REPO" /app/autogluon-bench \
        && pip install -e /app/autogluon-bench; \
    else \
        output=$(pip install autogluon.bench==$AG_BENCH_VERSION 2>&1) || true; \
        if echo $output | grep -q "No matching distribution"; then \
            echo -e "ERROR: No matching distribution found for autogluon.bench==$AG_BENCH_VERSION\n\
            To resolve the issue, try 'agbench run <config_file> --dev-branch <autogluon_bench_fork_uri>#<git_branch>'"; \
            exit 1; \
        fi; \
    fi

RUN chmod +x setup.sh entrypoint.sh /app/gpu_utilization.sh \
    && if echo "$FRAMEWORK_PATH" | grep -q "tabular"; then \
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
    && echo "*/5 * * * * /app/gpu_utilization.sh >> /var/log/cron.log 2>&1" > /var/spool/cron/crontabs/root \
    && chmod 600 /var/spool/cron/crontabs/root \
    && echo "CDK_DEPLOY_REGION=$CDK_DEPLOY_REGION" >> /etc/environment

ENTRYPOINT ["./entrypoint.sh"]
