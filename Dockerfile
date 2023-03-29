FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.13.1-gpu-py39-cu117-ubuntu20.04-ec2

WORKDIR /app/
COPY . /app/


RUN pip install --upgrade pip \
    && pip install -e .

RUN chmod +x entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]