"""Insert 7 new Track A slides into the dual-track main deck.

Adds slides as full-bleed picture inserts, then reorders the sldIdLst XML so
each new slide lands at the right narrative position. Slide 11 (Preliminary
Results, wrong 0.704 number) is deleted and replaced by the new Privacy slide.

Run from project root:
    .venv/bin/python slide/scripts/build_enriched_pptx.py
"""

from copy import deepcopy
from pathlib import Path
from pptx import Presentation
from pptx.util import Emu

SLIDE_DIR = Path(__file__).resolve().parent.parent
SRC = SLIDE_DIR / "FaceVeil_Dual_Track_Anonymization v2 (main).pptx"
OUT = SLIDE_DIR / "FaceVeil_Dual_Track_Anonymization v2 (main) - enriched.pptx"
ASSETS = SLIDE_DIR / "assets"

# Source-deck positions are 1-indexed and refer to the ORIGINAL 12-slide deck.
# After processing, the final deck is 18 slides.
INSERTIONS = [
    # (png_filename, action, anchor_slide_in_original_deck)
    # action "before"  = insert before this original slide
    # action "delete"  = drop this original slide entirely (png ignored, pass None)
    # All 7 Track A slides bundled before original slide 7 (Track B intro)
    # so the presenter can sweep through Track A in one go and hand off cleanly.
    ("slide.001.png", "before",  6),   # Why Pure-Classical IVP?
    ("slide.002.png", "before",  6),   # Pipeline Overview
    ("slide.003.png", "before",  7),   # 5 Theme Palettes
    ("slide.004.png", "before",  7),   # Temporal Stability
    ("slide.005.png", "before",  7),   # Privacy — FaceNet Re-ID
    ("slide.006.png", "before",  7),   # Joint Trade-Off scatter
    ("slide.007.png", "before",  7),   # Limitations & Non-Claims
    (None,            "delete", 11),   # drop original Preliminary Results (wrong 0.704)
]


def add_full_bleed_slide(prs, png_path):
    """Append a blank slide with the PNG covering the full canvas."""
    blank = prs.slide_layouts[0]
    slide = prs.slides.add_slide(blank)
    # Remove any auto-added placeholders so the picture is the only content
    for ph in list(slide.placeholders):
        sp = ph._element
        sp.getparent().remove(sp)
    slide.shapes.add_picture(
        str(png_path),
        left=Emu(0), top=Emu(0),
        width=prs.slide_width, height=prs.slide_height,
    )
    return slide


def move_slide(prs, from_idx, to_idx):
    """Move slide at from_idx to to_idx in sldIdLst (0-based)."""
    sldIdLst = prs.slides._sldIdLst
    slides = list(sldIdLst)
    item = slides[from_idx]
    sldIdLst.remove(item)
    sldIdLst.insert(to_idx, item)


def delete_slide(prs, idx):
    """Remove slide at idx from sldIdLst (does not purge the part)."""
    sldIdLst = prs.slides._sldIdLst
    slides = list(sldIdLst)
    sldIdLst.remove(slides[idx])


def main():
    prs = Presentation(str(SRC))
    n_orig = len(prs.slides)
    print(f"Original deck: {n_orig} slides")

    # Step 1: append each new slide at the END of the deck. Remember the
    # original-deck anchor + action for each so we can reorder later.
    new_slide_indices = []  # (orig_anchor, action, appended_idx)
    for png, action, anchor in INSERTIONS:
        if action == "delete":
            new_slide_indices.append((anchor, action, None))
            print(f"  delete original slide {anchor}")
            continue
        add_full_bleed_slide(prs, ASSETS / png)
        new_slide_indices.append((anchor, action, len(prs.slides) - 1))
        print(f"  appended {png} -> idx {len(prs.slides) - 1} "
              f"({action} original slide {anchor})")

    # Step 2: compute target positions. Original slide i (1-indexed) stays at
    # its position; "before" insertions shift later slides down; "replace"
    # deletes the original then inserts in its slot.
    #
    # We walk through the original slide order, building a final order list of
    # tuples (kind, ref) where kind is "orig" (original-deck index, 1-based)
    # or "new" (appended-deck index, 0-based).
    final_order = []
    replaced = {a for (a, act, _) in new_slide_indices if act == "replace"}
    deleted = {a for (a, act, _) in new_slide_indices if act == "delete"}

    # Pre-bucket new slides by anchor + action, preserving INSERTIONS order
    before_by_anchor = {}
    replace_by_anchor = {}
    for anchor, action, appended_idx in new_slide_indices:
        if action == "before":
            before_by_anchor.setdefault(anchor, []).append(appended_idx)
        elif action == "replace":
            replace_by_anchor[anchor] = appended_idx
        # "delete" handled by the `deleted` set

    for orig in range(1, n_orig + 1):
        for new_idx in before_by_anchor.get(orig, []):
            final_order.append(("new", new_idx))
        if orig in deleted:
            continue
        if orig in replaced:
            final_order.append(("new", replace_by_anchor[orig]))
        else:
            final_order.append(("orig", orig - 1))  # 0-based

    print(f"\nFinal deck plan ({len(final_order)} slides):")
    for i, (kind, ref) in enumerate(final_order, 1):
        print(f"  {i:2d}. {kind:4s} {ref}")

    # Step 3: rebuild sldIdLst in this order.
    # Current sldIdLst layout = [orig 0..n_orig-1, new 0..len(new)-1]
    sldIdLst = prs.slides._sldIdLst
    current_items = list(sldIdLst)
    # Clear and re-append in the planned order
    for item in current_items:
        sldIdLst.remove(item)
    for kind, ref in final_order:
        # ref is already the absolute index into current_items for both kinds
        sldIdLst.append(current_items[ref])

    print(f"\nFinal slide count: {len(prs.slides)}")
    prs.save(str(OUT))
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
