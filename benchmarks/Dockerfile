From 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.12.1-gpu-py38-cu116-ubuntu20.04-ec2

ARG git_user
ARG git_branch
ARG module
ARG data_path
ARG benchmark_name


ENV GIT_USER=$git_user
ENV GIT_BRANCH=$git_branch
ENV MODULE=$module
ENV DATA_PATH=$data_path
ENV BENCHMARK_NAME=$benchmark_name

WORKDIR /app/
COPY . /app/

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

CMD ["python", "run_benchmark.py", "--git_user", "${GIT_USER}", "--git_branch", "${GIT_BRANCH}", "--module", "${MODULE}", "--data_path", "${DATA_PATH}", "--benchmark_name", "${BENCHMARK_NAME}"]
