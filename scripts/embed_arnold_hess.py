import argparse
import os
from pathlib import Path

import cv2
import dotenv

from watermarks.arnold_hess_adapter import embed_image
from watermarks import arnold_hess as core


dotenv.load_dotenv(override=False)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Embed Arnold-Hess images into official WAVES data folders.")
    parser.add_argument("--input", required=True, help="Folder containing 512x512 host images")
    parser.add_argument("--watermark", required=True, help="64x64 binary watermark image")
    parser.add_argument("--dataset", default="custom", help="WAVES dataset name")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    data_dir = os.environ.get("DATA_DIR")
    if not data_dir:
        raise RuntimeError("DATA_DIR is not set. Put it in .env or export it.")

    os.environ["ARNOLD_HESS_WM_PATH"] = args.watermark

    real_dir = os.path.join(data_dir, "main", args.dataset, "real")
    wm_dir = os.path.join(data_dir, "main", args.dataset, "arnold_hess")
    side_dir = os.path.join(data_dir, "side_info", args.dataset, "arnold_hess")
    for d in [real_dir, wm_dir, side_dir]:
        ensure_dir(d)

    image_paths = core.find_host_images(input_folder=args.input, watermark_path=args.watermark)
    if args.limit is not None:
        image_paths = image_paths[: int(args.limit)]
    if not image_paths:
        raise RuntimeError(f"No valid {core.HOST_SIZE}x{core.HOST_SIZE} images found in {args.input}")

    print(f"[INFO] Embedding {len(image_paths)} images")
    for idx, host_path in enumerate(image_paths):
        print(f"[{idx + 1}/{len(image_paths)}] {os.path.basename(host_path)}")
        real_out = os.path.join(real_dir, f"{idx}.png")
        wm_out = os.path.join(wm_dir, f"{idx}.png")
        side_out = os.path.join(side_dir, f"{idx}.npz")
        img = core.load_color_image_24bit_safe(host_path, size=core.HOST_SIZE)
        cv2.imwrite(real_out, img)
        embed_image(host_path, args.watermark, wm_out, side_out)

    print("[DONE]")
    print("Real images:", real_dir)
    print("Watermarked images:", wm_dir)
    print("Side info:", side_dir)


if __name__ == "__main__":
    main()
