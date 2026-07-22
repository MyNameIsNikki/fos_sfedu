"""Check that required packages and hardware are available."""

import sys
import importlib


REQUIRED = [
    "torch",
    "torchvision",
    "cv2",
    "numpy",
    "matplotlib",
    "sklearn",
    "transformers",
    "timm",
]

OPTIONAL = [
    "ultralytics",
    "segmentation_models_pytorch",
    "nltk",
    "levenshtein",
    "scipy",
    "captum",
    "umap",
]


def check(pkg_name: str, import_name: str | None = None) -> bool:
    try:
        importlib.import_module(import_name or pkg_name)
        return True
    except ImportError:
        return False


def main():
    print(f"Python: {sys.version}")
    print()

    all_ok = True
    for pkg in REQUIRED:
        ok = check(pkg)
        print(f"  {'[OK]' if ok else '[MISSING]'} {pkg}")
        all_ok &= ok

    print()
    for pkg in OPTIONAL:
        ok = check(pkg)
        print(f"  {'[OK]' if ok else '[—]'} {pkg} (optional)")

    print()
    if all_ok:
        try:
            import torch
            print(f"  PyTorch version: {torch.__version__}")
            print(f"  CUDA available:  {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"  CUDA device:     {torch.cuda.get_device_name(0)}")
                print(f"  CUDA version:    {torch.version.cuda}")
        except Exception as e:
            print(f"  Error checking PyTorch: {e}")

    print()
    if all_ok:
        print("All required packages are present.")
    else:
        print("Some required packages are missing — install with:")
        print("  pip install -r ../requirements-course.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
