"""Use Helvetica (regular, bold, oblique) in matplotlib figures on macOS.

Matplotlib registers only the first face of ``Helvetica.ttc``, so bold text
falls back to the regular face.  ``use_helvetica`` extracts the regular, bold,
oblique and bold-oblique faces to TrueType files (cached in the temp
directory), registers them, and sets text and mathtext to Helvetica.  Fonts are
embedded as TrueType (``pdf.fonttype`` 42), so text stays editable in the PDF.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib
from matplotlib import font_manager

COLLECTION = Path("/System/Library/Fonts/Helvetica.ttc")
FACES = {0: "Helvetica.ttf", 1: "Helvetica-Bold.ttf", 2: "Helvetica-Oblique.ttf", 3: "Helvetica-BoldOblique.ttf"}


def use_helvetica(base_size: float = 7.0) -> None:
    from fontTools.ttLib import TTCollection

    if not COLLECTION.exists():
        raise FileNotFoundError(f"{COLLECTION} not found; Helvetica is required for this figure")
    cache = Path(tempfile.gettempdir()) / "helvetica_faces"
    cache.mkdir(exist_ok=True)
    collection = None
    for index, name in FACES.items():
        path = cache / name
        if not path.exists():
            collection = collection or TTCollection(str(COLLECTION))
            collection.fonts[index].save(str(path))
        font_manager.fontManager.addfont(str(path))
    matplotlib.rcParams.update({
        "font.family": "Helvetica", "font.size": base_size, "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "custom", "mathtext.rm": "Helvetica", "mathtext.it": "Helvetica:italic",
        "mathtext.bf": "Helvetica:bold", "mathtext.default": "regular",
    })
