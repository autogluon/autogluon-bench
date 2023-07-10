FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.13.1-gpu-py39-cu117-ubuntu20.04-ec2
ARG AG_BENCH_DEV_URL
ARG AG_BENCH_VERSION

WORKDIR /app/
COPY . /app/

RUN pip install --upgrade pip
RUN if [ -n "$AG_BENCH_DEV_URL" ]; then \
        echo "Cloning: $AG_BENCH_DEV_URL" \
        && AG_BENCH_DEV_REPO=$(echo "$AG_BENCH_DEV_URL" | cut -d "#" -f 1) \
        && AG_BENCH_DEV_BRANCH=$(echo "$AG_BENCH_DEV_URL" | cut -d "#" -f 2) \
        && git clone --branch "$AG_BENCH_DEV_BRANCH" --single-branch "$AG_BENCH_DEV_REPO" /app/autogluon-bench \
        && pip install -e /app/autogluon-bench; \
    else \
        echo "Installing autogluon.bench==$AG_BENCH_VERSION" \
        && pip install autogluon.bench==$AG_BENCH_VERSION; \
    fi

RUN chmod +x entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]
