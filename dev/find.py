import os
import warnings

from .constants import DATASET_NAMES, WATERMARK_METHODS


def check_file_existence(path, name_pattern, limit):
    if not os.path.isdir(path):
        return [False for _ in range(limit)]
    found_filenames = set(os.listdir(path))
    return [name_pattern.format(i) in found_filenames for i in range(limit)]


def existence_operation(existences1, existences2, op):
    if op == "difference":
        return [a and not b for a, b in zip(existences1, existences2)]
    elif op == "union":
        # Historical WAVES name: this is actually logical AND.
        return [a and b for a, b in zip(existences1, existences2)]
    else:
        raise ValueError(f"Invalid operation {op}, can be 'difference' or 'union'")


def existence_to_indices(existences, limit):
    return [i for i in range(min(len(existences), limit)) if existences[i]]


def _valid_dataset_name(dataset_name):
    return dataset_name in DATASET_NAMES


def _valid_source_name(source_name):
    if source_name == "real":
        return True
    if source_name in WATERMARK_METHODS:
        return True
    if source_name.startswith("real_"):
        return source_name[len("real_") :] in WATERMARK_METHODS
    return False


def parse_image_dir_path(path, quiet=True):
    data_dir = os.environ.get("DATA_DIR")
    if data_dir is None:
        raise ValueError("DATA_DIR is not set")
    if not os.path.commonpath([data_dir, str(path)]) == os.path.commonpath([data_dir]):
        raise ValueError(f"Image directory should be under DATA_DIR={data_dir}")
    try:
        mode, dataset_name, dirname = str(path).rstrip("/").split("/")[-3:]
    except ValueError:
        raise ValueError("Invalid image directory path, unable to parse")

    if not _valid_dataset_name(dataset_name):
        raise ValueError(f"Dataset name must be one of {list(DATASET_NAMES.keys())}, found {dataset_name}")

    if mode == "attacked":
        if not len(dirname.split("-")) == 3:
            raise ValueError(f"Attack directory name {dirname} is not in 'attack_name-attack_strength-source_name' format")
        attack_name, attack_strength, source_name = dirname.split("-")
        try:
            attack_strength = float(attack_strength)
            if attack_strength <= 0:
                raise ValueError("Attack strength must be positive")
        except ValueError:
            raise ValueError("Attack strength must be a positive number")
        if not _valid_source_name(source_name):
            raise ValueError(f"Source name must be real, a watermark method, or real_<method>; found {source_name}")
        if not quiet:
            print(" -- Dataset name:", dataset_name)
            print(" -- Attack name:", attack_name)
            print(" -- Attack strength:", attack_strength)
            print(" -- Source name:", source_name)
        return dataset_name, attack_name, attack_strength, source_name

    if mode == "main":
        if not _valid_source_name(dirname) or dirname.startswith("real_"):
            raise ValueError(f"Main source name must be real or a watermark method; found {dirname}")
        source_name = dirname
        if not quiet:
            print(" -- Dataset name:", dataset_name)
            print(" -- Attack name:", None)
            print(" -- Attack strength:", None)
            print(" -- Source name:", source_name)
        return dataset_name, None, None, source_name

    raise ValueError("Invalid image directory path, unable to parse")


def get_all_image_dir_paths(criteria=None):
    if criteria is not None and not callable(criteria):
        raise ValueError("criteria must be callable")
    data_dir = os.environ.get("DATA_DIR")
    dir_paths = []
    for mode in ["main", "attacked"]:
        mode_dir = os.path.join(data_dir, mode)
        if not os.path.isdir(mode_dir):
            continue
        for dataset_name in os.listdir(mode_dir):
            dataset_dir = os.path.join(mode_dir, dataset_name)
            if not os.path.isdir(dataset_dir):
                continue
            for dirname in os.listdir(dataset_dir):
                path = os.path.join(dataset_dir, dirname)
                if os.path.isdir(path):
                    dir_paths.append(path)
    image_dir_dict = {}
    for path in dir_paths:
        try:
            key = parse_image_dir_path(path)
            if criteria is None or criteria(*key):
                image_dir_dict[key] = path
        except ValueError:
            warnings.warn(f"Found invalid image directory {path}, skipping")
    return image_dir_dict


def parse_json_path(path):
    result_dir = os.environ.get("RESULT_DIR")
    if result_dir is None:
        raise ValueError("RESULT_DIR is not set")
    if not os.path.commonpath([result_dir, str(path)]) == os.path.commonpath([result_dir]):
        raise ValueError(f"JSON files should be under RESULT_DIR={result_dir}")
    if not str(path).endswith(".json"):
        raise ValueError("Invalid JSON file path, must end with .json")

    dataset_name, filename = str(path).rstrip("/").split("/")[-2:]
    if not _valid_dataset_name(dataset_name):
        raise ValueError(f"Dataset name must be one of {list(DATASET_NAMES.keys())}, found {dataset_name}")

    if filename.count("-") == 1:
        attack_name, attack_strength, source_name, result_type = None, None, *str(filename[:-5]).split("-")
    elif filename.count("-") == 3:
        attack_name, attack_strength, source_name, result_type = str(filename[:-5]).split("-")
        try:
            attack_strength = float(attack_strength)
            if attack_strength <= 0:
                raise ValueError("Attack strength must be positive")
        except ValueError:
            raise ValueError("Attack strength must be a positive number")
    else:
        raise ValueError(f"Invalid JSON file name {filename}")

    if result_type not in ["status", "reverse", "decode", "metric"]:
        raise ValueError("Invalid result type")
    if source_name is not None and not _valid_source_name(source_name):
        raise ValueError(f"Invalid source name {source_name}")
    return dataset_name, attack_name, attack_strength, source_name, result_type


def get_all_json_paths(criteria=None):
    if criteria is not None and not callable(criteria):
        raise ValueError("criteria must be callable")
    result_dir = os.environ.get("RESULT_DIR")
    json_paths = []
    if not os.path.isdir(result_dir):
        return {}
    for dataset_name in os.listdir(result_dir):
        dataset_dir = os.path.join(result_dir, dataset_name)
        if not os.path.isdir(dataset_dir):
            continue
        for filename in os.listdir(dataset_dir):
            path = os.path.join(dataset_dir, filename)
            if os.path.isfile(path):
                json_paths.append(path)
    json_dict = {}
    for path in json_paths:
        try:
            key = parse_json_path(path)
            if criteria is None or criteria(*key):
                json_dict[key] = path
        except ValueError as e:
            if not path.endswith("prompts.json"):
                warnings.warn(f"Found invalid JSON file {path}, {e}, skipping")
    return json_dict
