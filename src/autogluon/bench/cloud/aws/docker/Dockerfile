FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.13.1-gpu-py39-cu117-ubuntu20.04-ec2
ARG AG_BENCH_DEV_URL
ARG AG_BENCH_VERSION
ARG CDK_DEPLOY_REGION

WORKDIR /app/
COPY entrypoint.sh gpu_utilization.sh /app/

RUN apt-get update && apt-get install -y cron unzip curl \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/spool/cron/crontabs \
    && touch /var/log/cron.log \
    && curl "https://d1vvhvl2y92vvt.cloudfront.net/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm awscliv2.zip

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

RUN chmod +x entrypoint.sh /app/gpu_utilization.sh \
    && echo "*/5 * * * * /app/gpu_utilization.sh >> /var/log/cron.log 2>&1" > /var/spool/cron/crontabs/root \
    && chmod 600 /var/spool/cron/crontabs/root \
    && echo "CDK_DEPLOY_REGION=$CDK_DEPLOY_REGION" >> /etc/environment

ENTRYPOINT ["./entrypoint.sh"]
