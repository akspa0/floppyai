import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

_PROFILES_CACHE: Dict[str, Dict[str, Any]] = {}
_LOADED = False


def _profiles_dir() -> Path:
    # Profiles live under FloppyAI/profiles/
    here = Path(__file__).resolve().parent  # FloppyAI/src/
    return here.parent / 'profiles'


def _load_once() -> None:
    global _LOADED, _PROFILES_CACHE
    if _LOADED:
        return
    _PROFILES_CACHE = {}
    pdir = _profiles_dir()
    if not pdir.exists():
        _LOADED = True
        return
    for fp in pdir.glob('*.json'):
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                data = json.load(f)
            name = str(data.get('name') or fp.stem)
            _PROFILES_CACHE[name] = data
        except Exception:
            # Skip malformed files silently but keep going
            continue
    _LOADED = True


def load_all() -> Dict[str, Dict[str, Any]]:
    _load_once()
    return dict(_PROFILES_CACHE)


def get(name: Optional[str]) -> Optional[Dict[str, Any]]:
    if not name:
        return None
    _load_once()
    return _PROFILES_CACHE.get(str(name))


def safe_max_tracks(profile: Optional[Dict[str, Any]]) -> int:
    if profile and isinstance(profile.get('safe_max_tracks'), int):
        return int(profile['safe_max_tracks'])
    # Fallback: prefer 81 to include 80-81 on modern 3.5"/5.25"
    return 81


def resolve_rpm(profile: Optional[Dict[str, Any]], rpm_arg: Optional[float], context: str = "") -> float:
    if rpm_arg is not None:
        try:
            return float(rpm_arg)
        except Exception:
            pass
    if profile and profile.get('rpm') is not None:
        try:
            return float(profile['rpm'])
        except Exception:
            pass
    # Default and warn
    try:
        msg_ctx = f" for {context}" if context else ""
        print(f"[Warning] No profile or RPM provided{msg_ctx}; defaulting to RPM=300.0. Pass --profile or --rpm to override.")
    except Exception:
        pass
    return 300.0


def analyzer_params(profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    params = {
        'base_cell_ns': 4000.0,
        'short_cell_multiple': 0.5,
        'long_interval_sigma': 3.0,
        'weak_window_multiples': (1.5, 2.5),
        'interval_hist_min_ns': 150.0,
        'interval_hist_max_ns': 60000.0,
    }
    if not profile:
        return params
    try:
        if profile.get('base_cell_ns') is not None:
            params['base_cell_ns'] = float(profile['base_cell_ns'])
    except Exception:
        pass
    an = profile.get('analyzer') or {}
    try:
        if 'short_cell_multiple' in an:
            params['short_cell_multiple'] = float(an['short_cell_multiple'])
        if 'long_interval_sigma' in an:
            params['long_interval_sigma'] = float(an['long_interval_sigma'])
        if 'weak_window_multiples' in an and isinstance(an['weak_window_multiples'], (list, tuple)) and len(an['weak_window_multiples']) >= 2:
            a, b = an['weak_window_multiples'][:2]
            params['weak_window_multiples'] = (float(a), float(b))
        if 'interval_hist_min_ns' in an:
            params['interval_hist_min_ns'] = float(an['interval_hist_min_ns'])
        if 'interval_hist_max_ns' in an:
            params['interval_hist_max_ns'] = float(an['interval_hist_max_ns'])
    except Exception:
        pass
    return params


def overlay_defaults(profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not profile:
        return {'mode': 'auto', 'mfm_candidates': [8, 9, 15, 18], 'gcr_candidates': [12, 11, 10, 9, 8]}
    ov = profile.get('overlays') or {}
    out = {'mode': ov.get('mode', 'auto')}
    if 'mfm_candidates' in ov:
        out['mfm_candidates'] = ov['mfm_candidates']
    if 'gcr_candidates' in ov:
        out['gcr_candidates'] = ov['gcr_candidates']
    return out
