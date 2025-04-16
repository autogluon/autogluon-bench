ARG AG_BENCH_BASE_IMAGE=nvidia/cuda:12.2.0-runtime-ubuntu20.04
FROM ${AG_BENCH_BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV RUNNING_IN_DOCKER=true

# Install essential packages and Python 3.9
# Python 3.10 and above causes issue for certain datasets https://github.com/pytorch/pytorch/issues/143288
RUN apt-get update && \
    apt-get install -y software-properties-common build-essential && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.9 python3.9-dev python3.9-distutils python3.9-venv ffmpeg libsm6 libxext6 && \
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
ARG AG_BENCH_BASE_DIR
ARG CDK_DEPLOY_REGION
ARG FRAMEWORK_PATH
ARG GIT_URI
ARG GIT_BRANCH
ARG AMLB_FRAMEWORK
ARG AMLB_USER_DIR

ENV AG_BENCH_BASE_DIR=${AG_BENCH_BASE_DIR}

WORKDIR /app/
ENV VENV_BASE_DIR=/app/venv

# Copying files for installing autogluon.bench from source, user needs to at the root of autogluon.bench repo as we need pyproject.toml
COPY . .
COPY ${AG_BENCH_BASE_DIR}/entrypoint.sh .
COPY ${AG_BENCH_BASE_DIR}/custom_configs custom_configs/

ENV VENV_PATH="$VENV_BASE_DIR/.venv"
RUN python3 -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install uv

# check if autogluon.bench version contains "dev" tag
RUN if echo "$AG_BENCH_VERSION" | grep -q "dev"; then \ 
        # install from local source
        python3 -m uv pip install .; \
    else \
        python3 -m uv pip install autogluon.bench==$AG_BENCH_VERSION; \
    fi

RUN chmod +x entrypoint.sh \
    && if echo "$FRAMEWORK_PATH" | grep -q -E "tabular|timeseries"; then \
        if [ -n "$AMLB_USER_DIR" ]; then \
            bash ${AG_BENCH_BASE_DIR}/${FRAMEWORK_PATH}/setup.sh $GIT_URI $GIT_BRANCH $VENV_BASE_DIR $AMLB_FRAMEWORK $AMLB_USER_DIR; \
        else \
            bash ${AG_BENCH_BASE_DIR}/${FRAMEWORK_PATH}/setup.sh $GIT_URI $GIT_BRANCH $VENV_BASE_DIR $AMLB_FRAMEWORK; \
        fi; \
    elif echo "$FRAMEWORK_PATH" | grep -q "multimodal"; then \
        bash ${AG_BENCH_BASE_DIR}/${FRAMEWORK_PATH}/setup.sh $GIT_URI $GIT_BRANCH $VENV_BASE_DIR $AG_BENCH_VERSION; \
    fi \
    && echo "CDK_DEPLOY_REGION=$CDK_DEPLOY_REGION" >> /etc/environment

ENTRYPOINT ["./entrypoint.sh"]
