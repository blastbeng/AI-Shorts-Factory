import os
import yaml

config_path = os.getenv("MODELS_CONFIG_PATH", "configs/models.yaml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

for category, models in config.items():
    if isinstance(models, dict):
        for model_name, model_info in models.items():
            if isinstance(model_info, dict) and "path" in model_info:
                path = model_info["path"]
                # Check if path exists (file or directory)
                if os.path.exists(path):
                    if os.path.isdir(path):
                        if os.listdir(path):  # Directory is not empty
                            model_info["status"] = "installed"
                            print(f"[OK] {category}/{model_name} marked as installed (found at {path}).")
                        else:
                            model_info["status"] = "not_installed"
                            print(f"[WARN] {category}/{model_name} directory is empty.")
                    else:
                        # It's a file
                        model_info["status"] = "installed"
                        print(f"[OK] {category}/{model_name} marked as installed (file found at {path}).")
                else:
                    model_info["status"] = "not_installed"
                    print(f"[INFO] {category}/{model_name} not found at {path}.")

with open(config_path, "w") as f:
    yaml.dump(config, f, default_flow_style=False)

print("\nModels status synced successfully.")
