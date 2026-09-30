"""Import isolated native Wan utilities without importing optional audio/I2V stacks."""

import importlib.util
import sys
import types
from pathlib import Path


def wan_module(root, name):
    directory = Path(root).resolve() / "wan" / "modules"
    package = "_pdd_native_wan"
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(directory)]
        sys.modules[package] = module
    full = package + "." + name
    if full not in sys.modules:
        spec = importlib.util.spec_from_file_location(full, directory / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[full] = module
        spec.loader.exec_module(module)
    return sys.modules[full]
