import yaml
import os
import datetime 
from tools import yaml_to_code

class Storage:
    def __init__(self):
        self.data = {}

    def store(self, key, value):
        self.data[key] = value
    
    def retrieve(self, key):
        assert key in self.data, f"Key {key} not found in storage"
        return self.data[key]
    
    def get(self, key, default=None):
        return self.data.get(key, default)
    
    def dump(self, path):
        with open(path, "w") as f:
            # store as yaml
            yaml.safe_dump(self.data, f)
    
    def load(self, path):
        with open(path, "r") as f:
            self.data = yaml.safe_load(f)

class Agent:
    def __init__(self, key, config: Storage):
        self.key = key
        self.config = config
        self.create_temp_dir()
    
    def run(self, storage: Storage):
        pass

    def create_temp_dir(self):
        config_data = self.config.retrieve(self.key)
        model_name = config_data["model_name"]
        temp_base_dir = config_data["temp_base_dir"]
        temp_dir = os.path.join(temp_base_dir, f"{datetime.datetime.now().strftime('%m%d%H%M%S')}", yaml_to_code.clean_model_name(model_name))
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        config_data["temp_dir"] = temp_dir
        return temp_dir

    def log_yaml_list(self, yaml_objs, prefix):
        temp_dir = self.config.retrieve(self.key)["temp_dir"]
        for (i, v) in enumerate(yaml_objs):
            store_path = os.path.join(temp_dir, f"{prefix}_{i}.yaml")
            with open(store_path, "w") as f:
                yaml.dump(yaml_to_code.convert_to_literal(v), f, default_flow_style=False, sort_keys=False, width=float("inf"))
    
    def log_py_list(self, py_strs, prefix):
        temp_dir = self.config.retrieve(self.key)["temp_dir"]
        for (i, v) in enumerate(py_strs):
            store_path = os.path.join(temp_dir, f"{prefix}_{i}.py")
            with open(store_path, "w") as f:
                f.write(v)
    
    def log_md_list(self, md_strs, prefix):
        temp_dir = self.config.retrieve(self.key)["temp_dir"]
        for (i, v) in enumerate(md_strs):
            store_path = os.path.join(temp_dir, f"{prefix}_{i}.md")
            with open(store_path, "w") as f:
                f.write(v)