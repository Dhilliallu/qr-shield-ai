import importlib
import platform
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"


def check_package(module_name: str) -> tuple[bool, str]:
    """Return whether a package imports successfully and its version or error."""
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "version unavailable")
        return True, str(version)
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"


def main() -> None:
    print("QR Shield AI - Environment Status")
    print("=" * 36)
    print(f"Python executable: {sys.executable}")
    print(f"Python: {platform.python_version()} ({platform.python_implementation()})")

    all_packages_available = True
    for module_name, display_name in (
        ("torch", "PyTorch"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
    ):
        available, details = check_package(module_name)
        status = "OK" if available else "NOT AVAILABLE"
        print(f"{display_name}: {status} ({details})")
        all_packages_available = all_packages_available and available

        if module_name == "torch" and available:
            torch = importlib.import_module("torch")
            cuda_available = torch.cuda.is_available()
            print(f"CUDA available: {'Yes' if cuda_available else 'No'}")
            if cuda_available:
                print(f"GPU: {torch.cuda.get_device_name(0)}")
        elif module_name == "torch":
            print("CUDA available: Not checked (PyTorch is unavailable)")

    dataset_available = DATASET_DIR.is_dir()
    print(
        "Dataset folder: "
        f"{'AVAILABLE' if dataset_available else 'NOT AVAILABLE'} ({DATASET_DIR})"
    )
    print("=" * 36)
    if all_packages_available and dataset_available:
        print("Environment check passed.")
    else:
        print("Environment check found missing packages or folders.")
        sys.exit(1)


if __name__ == "__main__":
    main()
