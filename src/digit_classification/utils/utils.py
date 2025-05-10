import yaml

def load_config(file_path="./configs/config.yaml"):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)  # Load YAML file safely
    return config