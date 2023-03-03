import json
import subprocess

G4_2 = "g4dn.2xlarge"
G4_4 = "g4dn.4xlarge"
G4_12 = "g4dn.12xlarge"
G5_4 = "g5.4xlarge"
P3_2 = "p3.2xlarge"
RESERVED_MEMORY_SIZE = 15000

gpu_map = {
    G4_2: 1,
    G4_4: 1,
    G4_12: 4,
    G5_4: 1,
    P3_2: 1,
}
vcpu_map = {
    G4_2: 8,
    G4_4: 16,
    G4_12: 48,
    G5_4: 16,
    P3_2: 8,
}
memory_map = {
    G4_2: 32000 - RESERVED_MEMORY_SIZE,
    G4_4: 64000 - RESERVED_MEMORY_SIZE,
    G4_12: 192000 - RESERVED_MEMORY_SIZE,
    G5_4: 64000 - RESERVED_MEMORY_SIZE,
    P3_2: 61000 - RESERVED_MEMORY_SIZE,
}
CONTEXT_FILE = "./cdk.context.json"

##### Infrastructure Parameters #####
PREFIX = "automm-test"
MAX_MACHINE_NUM = 20
BLOCK_DEVICE_VOLUME = 200
INSTANCE = G4_12
#####################################

# The context will be used to create AWS infrastracture stacks
CONTEXT_TO_PARSE = {
    "STACK_NAME_PREFIX": PREFIX,  # aws resource tag key, also used as name prefix for resources created
    "STACK_NAME_TAG": "benchmark",  # aws resource tag value
    "STATIC_RESOURCE_STACK_NAME": f"{PREFIX}-static-resource-stack",
    "BATCH_STACK_NAME": f"{PREFIX}-batch-stack",
    "EXPERIMENT_TABLE": f"{PREFIX}-bench-table",  # table to record training metrics
    "MODEL_BUCKET": f"{PREFIX}-models",  # bucket to upload trained model
    "DATA_BUCKET": "automl-mm-bench",  # bucket to download data
    "INSTANCE_TYPES": [
        INSTANCE
    ],  # can be a list of instance families or instance types
    "COMPUTE_ENV_MAXV_CPUS": vcpu_map[INSTANCE] * MAX_MACHINE_NUM,  # total max v_cpus in batch compute environment
    "CONTAINER_GPU": gpu_map[INSTANCE],  # GPU reserved for container
    "CONTAINER_VCPU": vcpu_map[INSTANCE],  # v_cpus reserved for container
    "CONTAINER_MEMORY": memory_map[INSTANCE],  # memory in MB reserved for container, also used for shm_size, i.e. `shared_memory_size`
    "BLOCK_DEVICE_VOLUME": BLOCK_DEVICE_VOLUME,  # device attached to instance, in GB
    "LAMBDA_FUNCTION_NAME": f"{PREFIX}-batch-job-function",
    "VPC_NAME": "automm-ap-bench-batch-stack/automm-ap-bench-vpc",  # it's recommended to share a vpc for all benchmark infra, you can lookup an existing VPC name under aws console -> VPC, if you want to create a new one, assign a new name
}


def update_context_config(file: str):
    with open(CONTEXT_FILE, "w+") as f:
        try:
            cdk_config = json.load(f)
        except:
            cdk_config = {}
        cdk_config.update(CONTEXT_TO_PARSE)
        json.dump(cdk_config, f, indent=2)
        f.close()

def deploy_stack():
    update_context_config(file=CONTEXT_FILE)
    subprocess.check_call(
        [
            "./scripts/deploy.sh",
            CONTEXT_TO_PARSE["STACK_NAME_PREFIX"],
            CONTEXT_TO_PARSE["STACK_NAME_TAG"],
            CONTEXT_TO_PARSE["STATIC_RESOURCE_STACK_NAME"],
            CONTEXT_TO_PARSE["BATCH_STACK_NAME"],
            str(CONTEXT_TO_PARSE["CONTAINER_MEMORY"]),
        ]
    )
