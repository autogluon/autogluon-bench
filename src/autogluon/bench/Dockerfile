ARG AG_BENCH_BASE_IMAGE
FROM $AG_BENCH_BASE_IMAGE

ENV DEBIAN_FRONTEND=noninteractive
ENV RUNNING_IN_DOCKER=true
ENV AGBENCH_BASE=src/autogluon/bench/

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
ARG AG_BENCH_VERSION
ARG CDK_DEPLOY_REGION
ARG FRAMEWORK_PATH
ARG GIT_URI
ARG GIT_BRANCH
ARG BENCHMARK_DIR
ARG AMLB_FRAMEWORK
ARG AMLB_USER_DIR

WORKDIR /app/

# Copying necessary files for autogluon-bench package
COPY . /app/

# check if autogluon.bench version contains "dev" tag
RUN if echo "$AG_BENCH_VERSION" | grep -q "dev"; then \ 
        # install from local source
        pip install /app/; \
    else \
        pip install autogluon.bench==$AG_BENCH_VERSION; \
    fi

COPY ${AGBENCH_BASE}custom_configs ${AGBENCH_BASE}entrypoint.sh /app/
RUN chmod +x entrypoint.sh \
    && if echo "$FRAMEWORK_PATH" | grep -q -E "tabular|timeseries"; then \
        if [ -n "$AMLB_USER_DIR" ]; then \
            bash ${AGBENCH_BASE}${FRAMEWORK_PATH}setup.sh $GIT_URI $GIT_BRANCH $BENCHMARK_DIR $AMLB_FRAMEWORK $AMLB_USER_DIR; \
        else \
            bash ${AGBENCH_BASE}${FRAMEWORK_PATH}setup.sh $GIT_URI $GIT_BRANCH $BENCHMARK_DIR $AMLB_FRAMEWORK; \
        fi; \
    elif echo "$FRAMEWORK_PATH" | grep -q "multimodal"; then \
        bash ${AGBENCH_BASE}${FRAMEWORK_PATH}setup.sh $GIT_URI $GIT_BRANCH $BENCHMARK_DIR $AG_BENCH_VERSION; \
    fi \
    && echo "CDK_DEPLOY_REGION=$CDK_DEPLOY_REGION" >> /etc/environment

ENTRYPOINT ["./entrypoint.sh"]
