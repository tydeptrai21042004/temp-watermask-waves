import os
import click
import torch
import numpy as np
try:
    import onnxruntime as ort
except Exception:
    ort = None
import torch.multiprocessing as mp
from PIL import Image, ImageOps
import warnings
from tqdm.auto import tqdm
import dotenv
from dev import (
    LIMIT,
    SUBSET_LIMIT,
    WATERMARK_METHODS,
    check_file_existence,
    existence_operation,
    existence_to_indices,
    parse_image_dir_path,
    save_json,
    load_json,
    encode_array_to_string,
)

try:
    from watermarks.arnold_hess_adapter import decode_message as decode_arnold_hess_message
except Exception:
    decode_arnold_hess_message = None

dotenv.load_dotenv(override=False)
warnings.filterwarnings("ignore")


def _result_json_path(path):
    return os.path.join(
        os.environ.get("RESULT_DIR"),
        str(path).rstrip("/").split("/")[-2],
        f"{str(path).rstrip('/').split('/')[-1]}-decode.json",
    )


def _arnold_side_info_dir(dataset_name):
    return os.path.join(os.environ.get("DATA_DIR"), "side_info", dataset_name, "arnold_hess")


def get_indices(mode, path, quiet, subset, limit, subset_limit):
    json_path = _result_json_path(path)
    if os.path.exists(json_path) and (data := load_json(json_path)) is not None:
        decoded_existences = [data.get(str(i), {}).get(mode) is not None for i in range(limit)]
        if (not subset and sum(decoded_existences) == limit) or (
            subset and sum(decoded_existences[:subset_limit]) == subset_limit
        ):
            return []
    else:
        decoded_existences = [False] * limit

    image_existences = check_file_existence(path, name_pattern="{}.png", limit=limit)
    reversed_latents_existences = check_file_existence(path, name_pattern="{}_reversed.pkl", limit=limit)

    if not quiet:
        print(f"Found {sum(image_existences)} images, and {sum(reversed_latents_existences)} reversed latents")

    if mode == "tree_ring":
        existences = reversed_latents_existences
    elif mode in ["stable_sig", "stegastamp"]:
        existences = image_existences
    elif mode == "arnold_hess":
        dataset_name, _, _, _source_name = parse_image_dir_path(path, quiet=True)
        side_dir = _arnold_side_info_dir(dataset_name)
        side_existences = check_file_existence(side_dir, name_pattern="{}.npz", limit=limit)
        existences = existence_operation(image_existences, side_existences, op="union")
        if not quiet:
            print(f"Found {sum(side_existences)} Arnold-Hess side-info files in {side_dir}")
    else:
        raise ValueError(f"Unknown decode mode: {mode}")

    if os.path.exists(json_path):
        existences = existence_operation(existences, decoded_existences, op="difference")

    return existence_to_indices(existences, limit=limit if not subset else subset_limit)


def init_model(mode, gpu):
    if mode == "tree_ring":
        size = 64
        radius = 10
        channel = 3
        mask = torch.zeros((1, 4, size, size), dtype=torch.bool)
        x0 = y0 = size // 2
        y, x = np.ogrid[:size, :size]
        y = y[::-1]
        mask[:, channel] = torch.tensor(((x - x0) ** 2 + (y - y0) ** 2) <= radius**2)
        return mask
    elif mode == "stable_sig":
        if ort is None:
            raise RuntimeError("onnxruntime is required for stable_sig decoding")
        if ort is None:
            raise RuntimeError("onnxruntime is required for stegastamp decoding")
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session_options.log_severity_level = 3
        return ort.InferenceSession(
            os.path.join(os.environ.get("MODEL_DIR"), "stable_signature.onnx"),
            providers=["CUDAExecutionProvider"],
            provider_options=[{"device_id": str(gpu)}],
            sess_options=session_options,
        )
    elif mode == "stegastamp":
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session_options.log_severity_level = 3
        return ort.InferenceSession(
            os.path.join(os.environ.get("MODEL_DIR"), "stega_stamp.onnx"),
            providers=["CUDAExecutionProvider"],
            provider_options=[{"device_id": str(gpu)}],
            sess_options=session_options,
        )
    elif mode == "arnold_hess":
        return None


def load_files(mode, path, indices):
    if mode == "tree_ring":
        return torch.cat([torch.load(os.path.join(path, f"{idx}_reversed.pkl"), map_location="cpu") for idx in indices], dim=0)
    elif mode == "stable_sig":
        return np.stack(
            [
                (((np.array(Image.open(os.path.join(path, f"{idx}.png")), dtype=np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).transpose((2, 0, 1)).astype(np.float32))
                for idx in indices
            ],
            axis=0,
        )
    elif mode == "stegastamp":
        return np.stack(
            [np.array(ImageOps.fit(Image.open(os.path.join(path, f"{idx}.png")), (400, 400)), dtype=np.float32) / 255.0 for idx in indices],
            axis=0,
        )
    elif mode == "arnold_hess":
        dataset_name, _, _, _source_name = parse_image_dir_path(path, quiet=True)
        side_dir = _arnold_side_info_dir(dataset_name)
        return [(os.path.join(path, f"{idx}.png"), os.path.join(side_dir, f"{idx}.npz")) for idx in indices]


def decode(mode, model, gpu, inputs):
    if mode == "tree_ring":
        fft_latents = torch.fft.fftshift(torch.fft.fft2(inputs.to(f"cuda:{gpu}")), dim=(-1, -2))
        messages = torch.stack([fft_latents[i].unsqueeze(0)[model].flatten() for i in range(fft_latents.shape[0])], dim=0)
        return torch.concatenate([messages.real, messages.imag], dim=1).cpu().numpy()
    elif mode == "stable_sig":
        outputs = model.run(None, {"image": inputs})
        return (outputs[0] > 0).astype(bool)
    elif mode == "stegastamp":
        outputs = model.run(None, {"image": inputs, "secret": np.zeros((inputs.shape[0], 100), dtype=np.float32)})
        return outputs[2].astype(bool)
    elif mode == "arnold_hess":
        if decode_arnold_hess_message is None:
            raise RuntimeError("watermarks.arnold_hess_adapter could not be imported")
        messages = []
        for image_path, embed_info_path in inputs:
            messages.append(decode_arnold_hess_message(image_path, embed_info_path).astype(bool))
        return messages


def worker(mode, gpu, path, indices, lock, counter, results):
    model = init_model(mode, gpu)
    batch_size = {"tree_ring": 32, "stable_sig": 4, "stegastamp": 4, "arnold_hess": 8}[mode]
    for it in range(0, len(indices), batch_size):
        batch_indices = indices[it : min(it + batch_size, len(indices))]
        inputs = load_files(mode, path, batch_indices)
        messages = decode(mode, model, gpu, inputs)
        with lock:
            counter.value += len(batch_indices)
            for idx, message in zip(batch_indices, messages):
                results[idx] = encode_array_to_string(message)


def process(mode, indices, path, quiet):
    if mode == "arnold_hess":
        results = {}
        batch_size = 8
        with tqdm(total=len(indices), desc="Decoding Arnold-Hess images", unit="file", disable=quiet) as pbar:
            for it in range(0, len(indices), batch_size):
                batch_indices = indices[it : min(it + batch_size, len(indices))]
                inputs = load_files(mode, path, batch_indices)
                messages = decode(mode, None, None, inputs)
                for idx, message in zip(batch_indices, messages):
                    results[idx] = encode_array_to_string(message)
                pbar.update(len(batch_indices))
        return results

    mp.set_start_method("spawn", force=True)
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available for processing")
    if not quiet:
        print(f"Using {num_gpus} GPUs for processing")

    num_workers = {"tree_ring": num_gpus, "stable_sig": num_gpus, "stegastamp": num_gpus * 2}[mode]
    chunk_size = max(1, len(indices) // num_workers)
    with mp.Manager() as manager:
        counter = manager.Value("i", 0)
        lock = manager.Lock()
        results = manager.dict()
        processes = []
        for rank in range(num_workers):
            start_idx = rank * chunk_size
            end_idx = None if rank == num_workers - 1 else (rank + 1) * chunk_size
            p = mp.Process(target=worker, args=(mode, rank % num_gpus, path, indices[start_idx:end_idx], lock, counter, results))
            p.start()
            processes.append(p)
        with tqdm(total=len(indices), desc="Decoding images or reversed latents", unit="file") as pbar:
            while True:
                with lock:
                    pbar.n = counter.value
                    pbar.refresh()
                    if counter.value >= len(indices):
                        break
        for p in processes:
            p.join()
        return dict(results)


def report(mode, path, results, quiet, limit):
    json_path = _result_json_path(path)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    if (not os.path.exists(json_path)) or (data := load_json(json_path)) is None:
        data = {}
        for idx in range(limit):
            data[str(idx)] = {_mode: results.get(idx) if mode == _mode else None for _mode in WATERMARK_METHODS.keys()}
    else:
        for idx in range(limit):
            data.setdefault(str(idx), {_mode: None for _mode in WATERMARK_METHODS.keys()})
        for idx, message in results.items():
            data[str(idx)][mode] = message
    save_json(data, json_path)
    if not quiet:
        print(f"Decoded messages saved to {json_path}")


def single_mode(mode, path, quiet, subset, limit, subset_limit):
    if mode not in WATERMARK_METHODS:
        raise ValueError(f"Unknown mode {mode}. Available: {list(WATERMARK_METHODS.keys())}")
    if not quiet:
        print(f"Decoding {mode} messages")
    indices = get_indices(mode, path, quiet, subset, limit, subset_limit)
    if len(indices) == 0:
        if not quiet:
            print("All messages requested already decoded")
        return
    results = process(mode, indices, path, quiet)
    report(mode, path, results, quiet, limit)


@click.command()
@click.option("--path", "-p", type=str, default=os.getcwd(), help="Path to image directory")
@click.option("--dry", "-d", is_flag=True, default=False, help="Dry run")
@click.option("--subset", "-s", is_flag=True, default=False, help="Run on subset only")
@click.option("--quiet", "-q", is_flag=True, default=False, help="Quiet mode")
@click.option("--mode", "mode_name", default=None, help="Decode only one watermark mode, e.g. arnold_hess")
@click.option("--limit", default=LIMIT, type=int, help="Number of expected images")
@click.option("--subset-limit", default=SUBSET_LIMIT, type=int, help="Subset size")
def main(path, dry, subset, quiet, mode_name, limit, subset_limit):
    _, _, _, source_name = parse_image_dir_path(path, quiet=quiet)
    if mode_name is not None:
        single_mode(mode_name, path, quiet, subset, limit, subset_limit)
        return
    if source_name == "real":
        for mode in WATERMARK_METHODS.keys():
            single_mode(mode, path, quiet, subset, limit, subset_limit)
        return
    for mode in WATERMARK_METHODS.keys():
        if source_name.endswith(mode):
            single_mode(mode, path, quiet, subset, limit, subset_limit)
            return
    raise ValueError(f"Invalid source name {source_name} encountered")


if __name__ == "__main__":
    main()
