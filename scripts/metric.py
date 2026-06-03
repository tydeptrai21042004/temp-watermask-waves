# Allow direct execution from the repository root with: python scripts/<name>.py
import sys
from pathlib import Path as _PathForSysPath
_PROJECT_ROOT = _PathForSysPath(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
import os
import click
import torch
import torch.multiprocessing as mp
from PIL import Image
import warnings
from tqdm.auto import tqdm
import dotenv
from dev import (
    LIMIT,
    SUBSET_LIMIT,
    QUALITY_METRICS,
    get_all_image_dir_paths,
    check_file_existence,
    existence_operation,
    existence_to_indices,
    parse_image_dir_path,
    save_json,
    load_json,
)

dotenv.load_dotenv(override=False)
warnings.filterwarnings("ignore")

DELTA_METRICS = ["aesthetics", "artifacts", "clip_score"]
PARALLEL_METRICS = []


def _metric_json_path(path):
    return os.path.join(
        os.environ.get("RESULT_DIR"),
        str(path).rstrip("/").split("/")[-2],
        f"{str(path).rstrip('/').split('/')[-1]}-metric.json",
    )


def get_indices(mode, path, clean_path, attacked_path, quiet, subset, limit, subset_limit):
    json_path = _metric_json_path(path)
    if os.path.exists(json_path) and (data := load_json(json_path)) is not None:
        if mode != "aesthetics_and_artifacts":
            measured_existences = [data.get(str(i), {}).get(mode) is not None for i in range(limit)]
        else:
            measured_existences = [
                data.get(str(i), {}).get("aesthetics") is not None and data.get(str(i), {}).get("artifacts") is not None
                for i in range(limit)
            ]
        if (not subset and sum(measured_existences) == limit) or (subset and sum(measured_existences[:subset_limit]) == subset_limit):
            return []
    else:
        measured_existences = [False] * limit

    clean_image_existences = check_file_existence(clean_path, name_pattern="{}.png", limit=limit)
    attacked_image_existences = check_file_existence(attacked_path, name_pattern="{}.png", limit=limit) if attacked_path is not None else [True] * limit
    if mode.endswith("_fid") and sum(clean_image_existences) != limit:
        raise ValueError(f"Cannot compute FID if not all {limit} clean images exist")
    if not quiet:
        if attacked_path is None:
            print(f"Found {sum(clean_image_existences)} not-attacked images")
        else:
            print(f"Found {sum(attacked_image_existences)} attacked images and {sum(clean_image_existences)} corresponding not-attacked images")
    existences = existence_operation(clean_image_existences, attacked_image_existences, op="union")
    if os.path.exists(json_path):
        existences = existence_operation(existences, measured_existences, op="difference")
    return existence_to_indices(existences, limit=limit if not subset else subset_limit)


def process_single(mode, indices, path, clean_path, attacked_path, quiet, limit):
    if mode.endswith("_fid"):
        from metrics.distributional import compute_fid
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("No GPUs available for FID processing")
        metric = float(compute_fid(clean_path, attacked_path, mode=mode.split("_")[0], device=torch.device("cuda"), batch_size=64, num_workers=8, verbose=not quiet))
        return {idx: metric for idx in existence_to_indices(check_file_existence(attacked_path, name_pattern="{}.png", limit=limit), limit=limit)}

    if mode in ["psnr", "ssim", "nmi"]:
        from metrics.image import compute_image_distance_repeated
        clean_images = [Image.open(os.path.join(clean_path, f"{idx}.png")) for idx in indices]
        attacked_images = [Image.open(os.path.join(attacked_path, f"{idx}.png")) for idx in indices]
        metrics = compute_image_distance_repeated(clean_images, attacked_images, metric_name=mode, num_workers=min(8, max(1, os.cpu_count() or 1)), verbose=not quiet)
        results = {idx: metric for idx, metric in zip(indices, metrics)}
        [image.close() for image in clean_images]
        [image.close() for image in attacked_images]
        return results

    if mode in ["lpips", "watson"]:
        from metrics.perceptual import load_perceptual_models, compute_perceptual_metric_repeated
        clean_images = [Image.open(os.path.join(clean_path, f"{idx}.png")) for idx in indices]
        attacked_images = [Image.open(os.path.join(attacked_path, f"{idx}.png")) for idx in indices]
        model = load_perceptual_models(mode, mode="alex" if mode == "lpips" else "dft", device=torch.device("cuda"))
        batch_size = 32
        metrics = []
        pbar = tqdm(total=len(indices), desc=f"Computing {mode} metrics", unit="image", disable=quiet)
        for it in range(0, len(indices), batch_size):
            metrics.extend(compute_perceptual_metric_repeated(clean_images[it : min(it + batch_size, len(indices))], attacked_images[it : min(it + batch_size, len(indices))], metric_name=mode, mode="alex" if mode == "lpips" else "dft", model=model, device=torch.device("cuda")))
            pbar.update(min(batch_size, len(indices) - it))
        pbar.close()
        results = {idx: metric for idx, metric in zip(indices, metrics)}
        [image.close() for image in clean_images]
        [image.close() for image in attacked_images]
        return results

    if mode == "aesthetics_and_artifacts":
        from metrics.aesthetics import load_aesthetics_and_artifacts_models, compute_aesthetics_and_artifacts_scores
        model = load_aesthetics_and_artifacts_models(device=torch.device("cuda"))
        images = [Image.open(os.path.join(path, f"{idx}.png")) for idx in indices]
        batch_size = 8
        metrics = []
        pbar = tqdm(total=len(indices), desc=f"Computing {mode} metrics", unit="image", disable=quiet)
        for it in range(0, len(indices), batch_size):
            aesthetics, artifacts = compute_aesthetics_and_artifacts_scores(images[it : min(it + batch_size, len(indices))], model, device=torch.device("cuda"))
            metrics.extend(list(zip(aesthetics, artifacts)))
            pbar.update(min(batch_size, len(indices) - it))
        pbar.close()
        results = {idx: metric for idx, metric in zip(indices, metrics)}
        [image.close() for image in images]
        return results

    if mode == "clip_score":
        from metrics.clip import load_open_clip_model_preprocess_and_tokenizer, compute_clip_score
        model = load_open_clip_model_preprocess_and_tokenizer(device=torch.device("cuda"))
        images = [Image.open(os.path.join(path, f"{idx}.png")) for idx in indices]
        prompts = list(load_json(os.path.join(os.environ.get("RESULT_DIR"), str(path).rstrip("/").split("/")[-2], "prompts.json")).values())
        prompts = [prompts[idx] for idx in indices]
        batch_size = 8
        metrics = []
        pbar = tqdm(total=len(indices), desc=f"Computing {mode}", unit="image", disable=quiet)
        for it in range(0, len(indices), batch_size):
            metrics.extend(compute_clip_score(images[it : min(it + batch_size, len(indices))], prompts[it : min(it + batch_size, len(indices))], model, device=torch.device("cuda")))
            pbar.update(min(batch_size, len(indices) - it))
        pbar.close()
        results = {idx: metric for idx, metric in zip(indices, metrics)}
        [image.close() for image in images]
        return results

    raise ValueError(f"Unknown metric mode {mode}")


def report(mode, path, results, quiet, limit):
    json_path = _metric_json_path(path)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    if (not os.path.exists(json_path)) or (data := load_json(json_path)) is None:
        data = {}
        for idx in range(limit):
            data[str(idx)] = {_mode: results.get(idx) if mode == _mode else None for _mode in QUALITY_METRICS.keys()}
    else:
        for idx in range(limit):
            data.setdefault(str(idx), {_mode: None for _mode in QUALITY_METRICS.keys()})
        for idx, metric in results.items():
            data[str(idx)][mode] = metric
    save_json(data, json_path)
    if not quiet:
        print(f"Computed {mode} metrics saved to {json_path}")


def single_mode(mode, path, clean_path, attacked_path, quiet, subset, limit, subset_limit):
    if not quiet:
        print(f"Computing {mode} metrics")
    indices = get_indices(mode, path, clean_path, attacked_path, quiet, subset, limit, subset_limit)
    if len(indices) == 0:
        if not quiet:
            print(f"All {mode} metrics requested already computed")
        return
    results = process_single(mode, indices, path, clean_path, attacked_path, quiet, limit)
    if mode == "aesthetics_and_artifacts":
        report("aesthetics", path, {idx: result[0] for idx, result in results.items()}, quiet, limit)
        report("artifacts", path, {idx: result[1] for idx, result in results.items()}, quiet, limit)
    else:
        report(mode, path, results, quiet, limit)


@click.command()
@click.option("--path", "-p", type=str, default=os.getcwd(), help="Path to image directory")
@click.option("--dry", "-d", is_flag=True, default=False, help="Dry run")
@click.option("--subset", "-s", is_flag=True, default=False, help="Run on subset only")
@click.option("--quiet", "-q", is_flag=True, default=False, help="Quiet mode")
@click.option("--metrics", "metrics_csv", default=None, help="Comma-separated metrics to compute, e.g. psnr,ssim,nmi")
@click.option("--limit", default=LIMIT, type=int, help="Number of expected images")
@click.option("--subset-limit", default=SUBSET_LIMIT, type=int, help="Subset size")
def main(path, dry, subset, quiet, metrics_csv, limit, subset_limit):
    dataset_name, attack_name, attack_strength, source_name = parse_image_dir_path(path, quiet=quiet)

    if attack_name is None or attack_strength is None:
        if not quiet:
            print(f"Only delta metrics {DELTA_METRICS} are required for clean images")
        modes = list(DELTA_METRICS)
        clean_path = path
        attacked_path = None
    else:
        if not quiet:
            print(f"Computing metrics for attacked images")
        clean_path_dict = get_all_image_dir_paths(
            lambda _dataset_name, _attack_name, _attack_strength, _source_name: (
                _dataset_name == dataset_name
                and _attack_name is None
                and _attack_strength is None
                and _source_name == (source_name if not source_name.startswith("real") else "real")
            )
        )
        if len(clean_path_dict) != 1:
            raise ValueError(f"Found {len(clean_path_dict)} corresponding clean image directories, expected exactly 1")
        modes = list(QUALITY_METRICS.keys())
        clean_path = list(clean_path_dict.values())[0]
        attacked_path = path
        if not quiet:
            print(f"Found corresponding clean directory: {clean_path}")

    if metrics_csv:
        requested = [m.strip() for m in metrics_csv.split(",") if m.strip()]
        modes = requested

    if "aesthetics" in modes or "artifacts" in modes:
        modes = [m for m in modes if m not in ["aesthetics", "artifacts", "clip_score"]] + ["aesthetics_and_artifacts"] + (["clip_score"] if "clip_score" in modes else [])

    for mode in modes:
        single_mode(mode, path, clean_path, attacked_path, quiet, subset, limit, subset_limit)


if __name__ == "__main__":
    main()
