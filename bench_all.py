import random
n_experiments = 5
seeds = []
for i in range(n_experiments):
    seeds.append(random.randint(0, 100))

# seeds = [22, 92, 54, 86, 41]

config_paths = [
    "sample_configs/paper_text_tabular_local_configs.yaml",
    "sample_configs/paper_text_local_configs.yaml",
    "sample_configs/paper_image_local_configs.yaml",
]
frameworks = [
    # "AutoGluon_best_master",
    # "ablation_base",
    # "ablation_greedy_soup",
    # "ablation_gradient_clip",
    "ablation_warmup_steps",
    "ablation_cosine_decay",
    "ablation_weight_decay",
    # "ablation_lr_decay",
    # "autokeras_master",
]
constraints = [
    "g4_12x"
]
# module = "autokeras"
module = "multimodal"

import yaml
import os
import subprocess

config_root = "./temp_configs"
os.makedirs(config_root, exist_ok=True)

for seed in seeds:
    print("Seed: ", seed)
    for constraint in constraints:
        os.makedirs(f"{config_root}/{constraint}", exist_ok=True)
        for framework in frameworks:
            # for shot in fs:
                config_dir = f"{config_root}/{constraint}/{framework}"
                os.makedirs(config_dir, exist_ok=True)

                for config_path in config_paths:
                    with open(config_path, "r") as f:
                        configs = yaml.safe_load(f)
                        if constraint == "g4_12x":
                            configs["cdk_context"]["PREFIX"] = f"{configs['cdk_context']['PREFIX']}-multi"
                        configs["constraint"] = constraint
                        configs["framework"] = framework
                        configs["module"] = module
                        configs["seed"] = seed 
                        # configs["custom_dataloader"]["shot"] = shot
                        configs["benchmark_name"] = f"{configs['benchmark_name']}-{seed}"
                        new_config_path = os.path.join(config_dir, os.path.basename(config_path))
                        with open(new_config_path, "w") as new_f:
                            yaml.dump(configs, new_f)
                        print("Running config: ", new_config_path)
                        command = ["agbench", "run", new_config_path]
                        subprocess.run(command)

