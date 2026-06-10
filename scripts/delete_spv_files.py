from pathlib import Path
import sys


def delete_spv_files(root):
    count = 0
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".spv":
            path.unlink()
            print(f"deleted {path}")
            count += 1
    return count


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    root = Path(sys.argv[1]) if sys.argv[1:] else project_root / "data" / "shaders"

    if not root.exists():
        raise SystemExit(f"path does not exist: {root}")

    count = delete_spv_files(root)
    print(f"deleted {count} spv files")
