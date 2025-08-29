from .config import Config

def load_config(config_path=None):
    return Config(config_path)
