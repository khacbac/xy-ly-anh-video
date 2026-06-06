"""Replace `ppt/media/image3.png` (the slide-5 Track A diagram) in the enriched
deck with the corrected `slide_classical_pipeline.png`.

Reads:
  slide/FaceVeil_Dual_Track_Anonymization v2 (main) - enriched.pptx
  slide/assets/slide_classical_pipeline.png

Writes:
  slide/FaceVeil_Dual_Track_Anonymization v2 (main) - enriched.pptx
    (in-place; original is backed up to `... - enriched.before-slide5-fix.pptx`)
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

SLIDE_DIR = Path(__file__).resolve().parent.parent
PPTX = SLIDE_DIR / "FaceVeil_Dual_Track_Anonymization v2 (main) - enriched.pptx"
BACKUP = SLIDE_DIR / "FaceVeil_Dual_Track_Anonymization v2 (main) - enriched.before-slide5-fix.pptx"
NEW_IMG = SLIDE_DIR / "assets" / "slide_classical_pipeline.png"
TARGET_MEMBER = "ppt/media/image3.png"


def main() -> int:
    if not PPTX.exists():
        raise SystemExit(f"missing pptx: {PPTX}")
    if not NEW_IMG.exists():
        raise SystemExit(
            f"missing image: {NEW_IMG} — run slide/scripts/render_classical_pipeline_slide.py first"
        )

    if not BACKUP.exists():
        shutil.copy2(PPTX, BACKUP)
        print(f"  [backup] {BACKUP.name}")

    tmp = PPTX.with_suffix(".tmp.pptx")
    new_bytes = NEW_IMG.read_bytes()

    with zipfile.ZipFile(PPTX, "r") as src, zipfile.ZipFile(
        tmp, "w", zipfile.ZIP_DEFLATED
    ) as dst:
        replaced = False
        for item in src.infolist():
            data = new_bytes if item.filename == TARGET_MEMBER else src.read(item.filename)
            if item.filename == TARGET_MEMBER:
                replaced = True
            dst.writestr(item, data)
        if not replaced:
            tmp.unlink(missing_ok=True)
            raise SystemExit(f"target not found in archive: {TARGET_MEMBER}")

    tmp.replace(PPTX)
    print(f"  [write] {PPTX.name}  ({TARGET_MEMBER} replaced, {len(new_bytes)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
