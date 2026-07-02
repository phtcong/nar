"""Pipeline visualization of the NAR process on a table image.

For each document, produces one figure with four panels, left to right:

  1. RT-DETR        - the reference model's predicted cells (orange).
  2. ConRTF         - the primary model's predicted cells (blue).
  3. TLR            - both overlaid; the frame and title give the table-level
                      accept/reject decision from their agreement (TLR sim).
  4. CLR            - if the table is accepted, the primary model's cells
                      colored green (kept by consensus) or red (rejected).
                      If the table is rejected by TLR, the table is abstained.

Usage:
    python demo/nar_viz.py \
        --gt_dir <gt> --pred_one_dir <rtdetr> --pred_two_dir <conrtf> \
        --images_dir <images> --out_dir <viz> [--tau 0.9] [--clr_threshold 1.0]
"""
import argparse
import os
import sys

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from congrid_io import load_congrid_data  # noqa: E402
from nar_pipeline import make_evaluator, tlr_similarity  # noqa: E402

GREEN = (0, 160, 0)
RED = (210, 0, 0)
ORANGE = (235, 140, 0)
BLUE = (0, 90, 230)
MAGENTA = (200, 0, 200)
GRAY = (140, 140, 140)
BAR_H = 18
GAP = 10


def _duplicate_cells(pred):
    """Indices of cells whose row duplicates an earlier row (same y-band).

    A model that emits two rows over the same band (e.g. a header detected
    twice) produces spurious overlapping cells. Detected from one model alone.
    """
    rows = pred["rows"]
    dup_rows = set()
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            ay0, ay1, by0, by1 = rows[i][1], rows[i][3], rows[j][1], rows[j][3]
            inter = max(0.0, min(ay1, by1) - max(ay0, by0))
            uni = max(ay1, by1) - min(ay0, by0)
            if uni > 0 and inter / uni > 0.6:
                dup_rows.add(j)
    idx = [k for k, c in enumerate(pred["cells"]) if c.get("row") in dup_rows]
    return idx, len(dup_rows)


def clr_keep_mask(p1, p2, clr_threshold):
    ev = make_evaluator()
    ev.setData(p1, p2)
    scores = ev.evaluate()["cell_scores"]
    return [float(s) >= clr_threshold for s in scores]


def _draw_cells(im, cells, colors, width=2):
    d = ImageDraw.Draw(im)
    for color, cell in zip(colors, cells):
        x0, y0, x1, y1 = [float(v) for v in cell["bbox"]]
        if x1 > x0 and y1 > y0:
            d.rectangle([x0, y0, x1, y1], outline=color, width=width)


INSET = 3  # shrink dropped cells so duplicates stacked on kept cells stay visible


def _emphasize_dropped(base, cells, mask):
    """Kept cells faint green; dropped cells bold red, filled and inset so that
    duplicate detections overlapping a kept cell remain visible."""
    im = base.convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    for keep, c in zip(mask, cells):
        if keep:
            continue
        x0, y0, x1, y1 = [float(v) for v in c["bbox"]]
        x0, y0, x1, y1 = x0 + INSET, y0 + INSET, x1 - INSET, y1 - INSET
        if x1 > x0 and y1 > y0:
            od.rectangle([x0, y0, x1, y1], fill=(210, 0, 0, 110))
    im = Image.alpha_composite(im, ov).convert("RGB")
    d = ImageDraw.Draw(im)
    for keep, c in zip(mask, cells):
        x0, y0, x1, y1 = [float(v) for v in c["bbox"]]
        if keep:
            if x1 > x0 and y1 > y0:
                d.rectangle([x0, y0, x1, y1], outline=(0, 150, 0), width=1)
        else:
            x0, y0, x1, y1 = x0 + INSET, y0 + INSET, x1 - INSET, y1 - INSET
            if x1 > x0 and y1 > y0:
                d.rectangle([x0, y0, x1, y1], outline=RED, width=3)
    return im


def _titled(im, title, bar_color, frame=None):
    w, h = im.size
    out = Image.new("RGB", (w, h + BAR_H), (255, 255, 255))
    out.paste(im, (0, BAR_H))
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, w, BAR_H], fill=bar_color)
    d.text((4, 4), title, fill=(255, 255, 255))
    if frame is not None:
        d.rectangle([0, BAR_H, w - 1, h + BAR_H - 1], outline=frame, width=3)
    return out


def _hconcat(panels):
    width = sum(p.width for p in panels) + GAP * (len(panels) - 1)
    height = max(p.height for p in panels)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.width + GAP
    return canvas


def render_pipeline(name, p1, p2, image_path, tau, clr_threshold):
    sim = tlr_similarity(p1, p2)
    accepted = sim >= tau
    mask = clr_keep_mask(p1, p2, clr_threshold)
    base = Image.open(image_path).convert("RGB")

    # 1. RT-DETR
    a = base.copy()
    _draw_cells(a, p1["cells"], [ORANGE] * len(p1["cells"]))
    panel1 = _titled(a, "1. RT-DETR (%d rows, %d cells)" % (len(p1["rows"]), len(p1["cells"])), ORANGE)

    # 2. ConRTF, with any duplicate-row over-detections flagged in magenta
    b = base.copy()
    _draw_cells(b, p2["cells"], [BLUE] * len(p2["cells"]))
    dup_idx, n_dup = _duplicate_cells(p2)
    if dup_idx:
        d = ImageDraw.Draw(b)
        for k in dup_idx:
            x0, y0, x1, y1 = [float(v) for v in p2["cells"][k]["bbox"]]
            x0, y0, x1, y1 = x0 + INSET, y0 + INSET, x1 - INSET, y1 - INSET
            if x1 > x0 and y1 > y0:
                d.rectangle([x0, y0, x1, y1], outline=MAGENTA, width=3)
    title2 = "2. ConRTF (%d rows, %d cells)" % (len(p2["rows"]), len(p2["cells"]))
    if n_dup:
        title2 += " - %d duplicate row" % n_dup
    panel2 = _titled(b, title2, BLUE)

    # 3. TLR: both overlaid, table-level decision
    c = base.copy()
    _draw_cells(c, p1["cells"], [ORANGE] * len(p1["cells"]), width=1)
    _draw_cells(c, p2["cells"], [BLUE] * len(p2["cells"]), width=1)
    frame = GREEN if accepted else RED
    status = "ACCEPT" if accepted else "REJECT"
    panel3 = _titled(c, "3. TLR sim=%.2f -> %s" % (sim, status), frame, frame=frame)

    # 4. CLR: consensus result on the primary model
    if accepted:
        e = _emphasize_dropped(base, p2["cells"], mask)
        kept = sum(mask)
        dropped = len(mask) - kept
        panel4 = _titled(e, "4. CLR: kept %d, dropped %d (no consensus)" % (kept, dropped), GREEN, frame=GREEN)
    else:
        e = base.copy()
        _draw_cells(e, p2["cells"], [GRAY] * len(p2["cells"]))
        panel4 = _titled(e, "4. CLR - table abstained", GRAY, frame=RED)

    return _hconcat([panel1, panel2, panel3, panel4])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--pred_one_dir", required=True, help="reference model (RT-DETR-X)")
    ap.add_argument("--pred_two_dir", required=True, help="primary/kept model (ConRTF)")
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--clr_threshold", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    names = sorted(f[:-5] for f in os.listdir(args.pred_two_dir) if f.endswith(".json"))
    for name in names:
        image_path = os.path.join(args.images_dir, name + ".jpg")
        if not os.path.exists(image_path):
            print("skip %s (no image)" % name)
            continue
        p1 = load_congrid_data(os.path.join(args.pred_one_dir, name + ".json"))
        p2 = load_congrid_data(os.path.join(args.pred_two_dir, name + ".json"))
        img = render_pipeline(name, p1, p2, image_path, args.tau, args.clr_threshold)
        out = os.path.join(args.out_dir, name + ".png")
        img.save(out)
        print("wrote %s" % out)


if __name__ == "__main__":
    main()
