import yaml
import os
from pathlib import Path


def get_absolute_path(relative_path: str) -> str:
    """
    Returns the absolute path of a given relative path.
    Works on Linux, macOS, and Windows.
    """
    return os.path.abspath(os.path.expanduser(relative_path))


def go_up_directories(abs_path: str, levels: int = 1) -> str:
    """
    Go up 'levels' directories from the given absolute path.
    """
    path = Path(abs_path).resolve()
    for _ in range(levels):
        path = path.parent
    return str(path)


def recursive_replace(data, old_value, new_value):
    """
    Recursively replace string values in nested dictionaries/lists.
    """
    if isinstance(data, dict):
        return {
            key: recursive_replace(value, old_value, new_value)
            for key, value in data.items()
        }
    elif isinstance(data, list):
        return [recursive_replace(item, old_value, new_value) for item in data]
    elif isinstance(data, str):
        return data.replace(old_value, new_value)
    else:
        return data


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "settings.yaml")

with open(CONFIG_PATH, "r") as f:
    settings = yaml.safe_load(f)
    ROOT_PATH = go_up_directories(Path(CONFIG_PATH).parent, 1)
    settings = recursive_replace(settings, old_value="<ROOT_PATH>", new_value=ROOT_PATH)
    settings["ROOT_PATH"] = ROOT_PATH
