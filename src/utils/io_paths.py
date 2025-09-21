import datetime
from pathlib import Path
import re

def get_output_dir(output_dir=None) -> Path:
    """Resolve output directory.

    - If output_dir is None: create test_outputs/<timestamp>/
    - If output_dir is provided: use it directly (no timestamp subfolder)
    """
    if output_dir is None:
        base_dir = Path("test_outputs")
        base_dir.mkdir(exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = base_dir / timestamp
        run_dir.mkdir(exist_ok=True)
        return run_dir
    else:
        run_dir = Path(output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir


def safe_label_from_path(p: Path) -> str:
    label = p.stem if p.is_file() else p.name
    return re.sub(r'[^A-Za-z0-9_.-]', '_', label)
