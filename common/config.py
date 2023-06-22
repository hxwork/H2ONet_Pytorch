import json
from easydict import EasyDict as edict


class Config():
    def __init__(self, json_path):
        with open(json_path) as f:
            self.cfg = json.load(f)
            self.cfg = edict(self.cfg)

    def save(self, json_path):
        with open(json_path, "w") as f:
            json.dump(self.cfg, f, indent=4)

    # @property
    # def dict(self):
    #     """Gives dict-like access to Params instance by `cfg.dict["learning_rate"]"""
    #     return self.cfg

    # def __call__(self):
    #     return self.cfg
