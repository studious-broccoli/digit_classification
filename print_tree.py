import os

EXCLUDE_DIRS = {'.git', '.idea', '__pycache__'}

def print_tree(start_path, indent=""):
    for item in sorted(os.listdir(start_path)):
        if item in EXCLUDE_DIRS:
            continue
        path = os.path.join(start_path, item)
        print(indent + "├── " + item)
        if os.path.isdir(path):
            print_tree(path, indent + "│   ")


print_tree("./")

