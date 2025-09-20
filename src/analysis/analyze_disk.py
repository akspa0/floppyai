from importlib import import_module

def run(args):
    """Temporary shim that delegates to main.analyze_disk(args).
    This allows us to rewire the CLI now, and move the actual implementation later
    without breaking the interface.
    """
    try:
        m = import_module('FloppyAI.src.main')
    except Exception:
        # Fallback for direct src execution where package name may differ
        m = import_module('main')
    return m.analyze_disk(args)
