import argparse
import dotenv

dotenv.load_dotenv(override=False)

from dev import get_performance, get_quality


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="custom")
    parser.add_argument("--method", default="arnold_hess")
    parser.add_argument("--attacks", nargs="+", default=["distortion_single_jpeg"])
    parser.add_argument("--strengths", nargs="+", default=["0.5"])
    parser.add_argument("--quality-mode", default="removal")
    args = parser.parse_args()

    for attack in args.attacks:
        for strength in args.strengths:
            strength_f = float(strength)
            perf = get_performance(args.dataset, args.method, attack, strength_f, mode="removal")
            qual = get_quality(args.dataset, args.method, attack, strength_f, mode=args.quality_mode)
            print("=" * 80)
            print(f"{attack} strength={strength}")
            print("Performance:", perf)
            print("Quality:", qual)


if __name__ == "__main__":
    main()
