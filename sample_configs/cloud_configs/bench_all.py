import random
n_experiments = 5
seeds = []
for i in range(n_experiments):
    seeds.append(random.randint(0, 100))
print(seeds)

seeds = [22, 92, 54, 86, 41]
config_paths = [
    "sample_configs/paper_text_tabular_local_configs.yaml",
    # "sample_configs/paper_text_local_configs.yaml",
    # "sample_configs/paper_image_local_configs.yaml",
    # "sample_configs/multimodal_cloud_text_configs.yaml",
    # "sample_configs/multimodal_cloud_text_fs_configs.yaml",
    # "sample_configs/multimodal_cloud_text_tabular_configs.yaml",
    # "sample_configs/multimodal_cloud_text_tabular_image_configs.yaml",
    # "sample_configs/multimodal_cloud_text_tabular_image_standard_configs.yaml"
]
frameworks = [
    # "AutoGluon_best_master",
    # "ablation_greedy_soup",
    # "ablation_gradient_clip",
    # "ablation_warmup_steps",
    # "ablation_cosine_decay",
    # "ablation_weight_decay",
    # "ablation_lr_decay",
    "autokeras_master",
    # "torch_compile_best",
    # "AutoGluon_best_master",
    # "AutoGluon_high_master",
    # "AutoGluon_medium_master",
    # "AutoGluon_high_vitlarge",
    # "AutoGluon_medium_vitlarge",
    # "AutoGluon_best_vitlarge",
    # "AutoGluon_best_caformer",
    # "AutoGluon_best_beit",
    # "AutoGluon_best_swinv2"
    # "AutoGluon_high_0_8",
    # "AutoGluon_medium_0_8",
    # "AutoGluon_best_0_8",
]
constraints = [
    "g4_12x"
]
fs = [
    1,
    5,
    10
]
module = "autokeras"
# module = "multimodal"

import yaml
import os
import subprocess

config_root = "./temp_configs"
os.makedirs(config_root, exist_ok=True)

for seed in seeds:
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
                        # configs["benchmark_name"] = f"{configs['benchmark_name']}-{shot}"
                        new_config_path = os.path.join(config_dir, os.path.basename(config_path))
                        with open(new_config_path, "w") as new_f:
                            yaml.dump(configs, new_f)

                        command = ["agbench", "run", new_config_path]
                        subprocess.run(command)
