from pathlib import Path
import json
import numpy as np

def _json_default(o):
    try:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
    except Exception:
        pass
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, set):
        return list(o)
    # Fallback stringification to avoid crashes
    return str(o)


def dump_json(path: Path, obj, *, indent: int = 2, allow_nan: bool | None = None):
    kwargs = {"indent": indent, "default": _json_default}
    if allow_nan is not None:
        kwargs["allow_nan"] = allow_nan
    with open(path, "w") as f:
        json.dump(obj, f, **kwargs)
