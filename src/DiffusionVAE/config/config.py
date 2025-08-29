import yaml
import os
from types import SimpleNamespace


class Config:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config.yml')

        self.config_path = config_path
        self._load_config()

    def _load_config(self):
        try:
            with open(self.config_path, 'r') as file:
                config_dict = yaml.safe_load(file)
            self._dict_to_namespace(config_dict)

        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML config: {e}")

    def _dict_to_namespace(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                namespace_obj = SimpleNamespace()
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, dict):
                        setattr(namespace_obj, sub_key, SimpleNamespace(**sub_value))
                    else:
                        setattr(namespace_obj, sub_key, sub_value)
                setattr(self, key, namespace_obj)
            else:
                setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if key.startswith('_') or key == 'config_path':
                continue
            if isinstance(value, SimpleNamespace):
                result[key] = self._namespace_to_dict(value)
            else:
                result[key] = value
        return result

    def _namespace_to_dict(self, namespace_obj):
        result = {}
        for key, value in namespace_obj.__dict__.items():
            if isinstance(value, SimpleNamespace):
                result[key] = self._namespace_to_dict(value)
            else:
                result[key] = value
        return result

    def __repr__(self):
        return f"Config(config_path='{self.config_path}')"

    def __str__(self):
        return yaml.dump(self.to_dict(), default_flow_style=False, indent=2)


def load_config(config_path=None):
    return Config(config_path)
