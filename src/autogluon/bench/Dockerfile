FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.13.1-gpu-py39-cu117-ubuntu20.04-ec2
ARG AG_BENCH_VERSION=latest

WORKDIR /app/
COPY . /app/


RUN pip install --upgrade pip \
    && pip install autogluon.bench==$AG_BENCH_VERSION

RUN chmod +x entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]
