#!/usr/bin/env python3
"""Fast, offline image check used at build time and before benchmark runs."""

from __future__ import annotations

import importlib
import json
import platform
import shutil
import sys


IMPORTS = [
    "numpy", "pandas", "scipy", "sklearn", "matplotlib", "PIL",
    "docx", "pptx", "openpyxl", "fitz", "soundfile", "librosa",
    "moviepy", "cv2", "reportlab",
    # Less common GDPval capabilities still fail preflight rather than being
    # misclassified later as model failures.
    "geopandas", "rasterio", "fiona", "shapely", "rdkit",
    "pedalboard", "weasyprint", "camelot", "trimesh", "mne", "Bio",
    "xgboost", "catboost", "lightgbm", "spacy", "h5py", "tables",
]
OPTIONAL_IMPORTS = ["cadquery", "aspose.words"]
BINARIES = [
    "bash", "python3", "ffmpeg", "ffprobe", "libreoffice", "pandoc",
    "pdftoppm", "tesseract", "dot", "node", "java", "R", "ruby",
]


def main() -> int:
    missing_imports = []
    for name in IMPORTS:
        try:
            importlib.import_module(name)
        except Exception as exc:  # import-time ABI failures matter too
            missing_imports.append({"name": name, "error": f"{type(exc).__name__}: {exc}"})
    missing_binaries = [name for name in BINARIES if shutil.which(name) is None]
    unavailable_optional = []
    for name in OPTIONAL_IMPORTS:
        try:
            importlib.import_module(name)
        except Exception:
            unavailable_optional.append(name)
    report = {
        "ok": not missing_imports and not missing_binaries,
        "architecture": platform.machine(),
        "python": sys.version.split()[0],
        "missing_imports": missing_imports,
        "missing_binaries": missing_binaries,
        "unavailable_optional_capabilities": unavailable_optional,
    }
    print(json.dumps(report, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
