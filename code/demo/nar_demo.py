"""NAR demo: Table-Level Rejection (TLR) + Cell-Level Rejection (CLR) + ConGrid.

Runs entirely on pre-computed ConGrid JSON. Two independently trained models
supply predictions. For each table:

  - TLR measures the agreement between the two models (ConGrid Cell-F1 of one
    vs the other). Tables below a threshold are rejected (table-level abstain).
  - CLR keeps only the cells where the two models agree (consensus) and scores
    those against ground truth.

The report shows each mechanism separately, mirroring the paper:

  1. Baseline           - primary model vs GT, no rejection.
  2. CLR only           - cell consensus on every table, no table rejection.
  3. TLR only (sweep)   - reject whole tables, accepted tables scored as-is.
  4. TLR + CLR (sweep)  - reject tables, then cell-consensus the rest.

Usage:
    python demo/nar_demo.py \
        --gt_dir       <dir of GT json> \
        --pred_one_dir <dir of RT-DETR json> \
        --pred_two_dir <dir of ConRTF json> \
        [--ignore_file image_ignore.txt] [--clr_threshold 1.0]

    python demo/nar_demo.py --selfcheck   # runs on the bundled fixture
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from congrid_io import load_congrid_data  # noqa: E402
from nar_pipeline import tlr_similarity, clr_score, baseline_score  # noqa: E402

DEFAULT_THRESHOLDS = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]


def _basenames(d):
    return {f[:-5] for f in os.listdir(d) if f.endswith(".json")}


def _read_ignore(path):
    if not path or not os.path.exists(path):
        return set()
    out = set()
    with open(path) as f:
        for line in f:
            name = line.strip()
            if name:
                out.add(os.path.splitext(name)[0])
    return out


def compute_per_table(gt_dir, pred_one_dir, pred_two_dir, ignore_file=None, clr_threshold=1.0):
    """Compute, once per table, its TLR similarity, baseline metrics (primary
    model vs GT) and CLR-refined metrics (consensus vs GT)."""
    ignore = _read_ignore(ignore_file)
    names = sorted((_basenames(gt_dir) & _basenames(pred_one_dir) & _basenames(pred_two_dir)) - ignore)
    rows = []
    for name in names:
        gt = load_congrid_data(os.path.join(gt_dir, name + ".json"))
        p1 = load_congrid_data(os.path.join(pred_one_dir, name + ".json"))
        p2 = load_congrid_data(os.path.join(pred_two_dir, name + ".json"))
        sim = tlr_similarity(p1, p2)
        clr = clr_score(gt, p1, p2, clr_threshold=clr_threshold)
        base = baseline_score(gt, p2)
        rows.append({
            "name": name, "tlr_sim": sim,
            "base_precision": base["cell_precision"], "base_recall": base["cell_recall"], "base_f1": base["cell_f1"],
            "clr_precision": clr["cell_precision"], "clr_recall": clr["cell_recall"], "clr_f1": clr["cell_f1"],
            "cell_retention": clr["cell_retention"],
        })
    return rows


def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def _agg(rows, prefix):
    return (_mean([r[prefix + "precision"] for r in rows]),
            _mean([r[prefix + "recall"] for r in rows]),
            _mean([r[prefix + "f1"] for r in rows]))


def sweep(per_table, thresholds, prefix):
    """Table-rejection sweep. prefix selects 'base_' (TLR only) or 'clr_' (TLR+CLR)."""
    total = len(per_table)
    out = []
    for tau in thresholds:
        acc = [r for r in per_table if r["tlr_sim"] >= tau]
        p, r, f = _agg(acc, prefix)
        out.append({"tau": tau, "coverage": (len(acc) / total) if total else 0.0,
                    "n_accepted": len(acc), "precision": p, "recall": r, "f1": f})
    return out


def err_delta(precision, base_precision):
    """Paper's primary metric: the share of the baseline's cell errors removed
    at a given operating point, ERR Δ = (precision - base) / (1 - base) * 100.

    %ERR = 100 - ConGrid-Cell precision, so this equals
    (%ERR_base - %ERR_here) / %ERR_base * 100. Returns None when the baseline
    has no cell errors (1 - base ≈ 0) and ERR Δ is undefined.
    """
    denom = 1.0 - base_precision
    if denom <= 1e-9:
        return None
    return (precision - base_precision) / denom * 100.0


def _fmt_errd(precision, base_precision):
    ed = err_delta(precision, base_precision)
    return "n/a" if ed is None else "%.1f%%" % ed


def _print_sweep(title, rows, err_baseline=None):
    """err_baseline: baseline precision. When given, append an ERR Δ column
    (only meaningful for CLR-based rows; omit it for the plain TLR sweep)."""
    show_err = err_baseline is not None
    print(title)
    if show_err:
        print("  %-6s %-9s %-9s %-10s %-9s %-9s %-9s" % (
            "tau", "coverage", "accepted", "precision", "recall", "f1", "ERR Δ"))
    else:
        print("  %-6s %-9s %-9s %-10s %-9s %-9s" % (
            "tau", "coverage", "accepted", "precision", "recall", "f1"))
    for row in rows:
        if row["n_accepted"] == 0:
            if show_err:
                print("  %-6.2f %-9.4f %-9d %-10s %-9s %-9s %-9s" % (
                    row["tau"], row["coverage"], 0, "n/a", "n/a", "n/a", "n/a"))
            else:
                print("  %-6.2f %-9.4f %-9d %-10s %-9s %-9s" % (
                    row["tau"], row["coverage"], 0, "n/a", "n/a", "n/a"))
            continue
        if show_err:
            print("  %-6.2f %-9.4f %-9d %-10.4f %-9.4f %-9.4f %-9s" % (
                row["tau"], row["coverage"], row["n_accepted"], row["precision"],
                row["recall"], row["f1"], _fmt_errd(row["precision"], err_baseline)))
        else:
            print("  %-6.2f %-9.4f %-9d %-10.4f %-9.4f %-9.4f" % (
                row["tau"], row["coverage"], row["n_accepted"], row["precision"], row["recall"], row["f1"]))


def print_per_table(per_table):
    print("Per-document detail:")
    print("  %-26s %-9s %-9s %-9s %-9s %-9s %-9s" % (
        "document", "tlr_sim", "base_pre", "clr_pre", "base_f1", "clr_f1", "retention"))
    for r in per_table:
        print("  %-26s %-9.4f %-9.4f %-9.4f %-9.4f %-9.4f %-9.4f" % (
            r["name"], r["tlr_sim"], r["base_precision"], r["clr_precision"],
            r["base_f1"], r["clr_f1"], r["cell_retention"]))
    print("")


def print_report(per_table, thresholds, clr_threshold, per_table_detail=False):
    total = len(per_table)
    if per_table_detail:
        print_per_table(per_table)
    bp, br, bf = _agg(per_table, "base_")
    cp, cr, cf = _agg(per_table, "clr_")
    retention = _mean([r["cell_retention"] for r in per_table])

    print("Tables: %d   CLR threshold: %.2f" % (total, clr_threshold))
    print("")
    print("1. Baseline (primary model vs GT, no rejection):")
    print("     precision=%.4f recall=%.4f f1=%.4f   %%ERR=%.2f%%" % (bp, br, bf, 100.0 * (1.0 - bp)))
    print("")
    print("2. CLR only (cell consensus on every table, table coverage=100%):")
    print("     precision=%.4f recall=%.4f f1=%.4f   mean cell retention=%.4f   ERR Δ=%s" % (
        cp, cr, cf, retention, _fmt_errd(cp, bp)))
    print("")
    _print_sweep("3. TLR only (reject tables, accepted scored as-is):", sweep(per_table, thresholds, "base_"))
    print("")
    _print_sweep("4. TLR + CLR (reject tables, then cell consensus):",
                 sweep(per_table, thresholds, "clr_"), err_baseline=bp)


def run_selfcheck():
    fx = os.path.join(HERE, "fixtures")
    per = compute_per_table(os.path.join(fx, "gt"), os.path.join(fx, "rtdetr"), os.path.join(fx, "conrtf"))
    print_report(per, DEFAULT_THRESHOLDS, 1.0)
    rows = sweep(per, DEFAULT_THRESHOLDS, "clr_")
    cov0 = rows[0]["coverage"]
    cov_last = rows[-1]["coverage"]
    prec0 = rows[0]["precision"]
    prec_high = max(r["precision"] for r in rows if r["n_accepted"] > 0)
    print("")
    assert abs(cov0 - 1.0) < 1e-9, "coverage at tau=0 should be 1.0, got %r" % cov0
    assert cov_last < 1.0, "coverage at tau=1.0 should drop below 1.0, got %r" % cov_last
    assert prec_high >= prec0 - 1e-9, "precision should not decrease as we reject, %r < %r" % (prec_high, prec0)
    print("SELFCHECK PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt_dir")
    ap.add_argument("--pred_one_dir", help="reference model (RT-DETR-X)")
    ap.add_argument("--pred_two_dir", help="primary/kept model (ConRTF)")
    ap.add_argument("--ignore_file", default=None)
    ap.add_argument("--clr_threshold", type=float, default=1.0)
    ap.add_argument("--thresholds", default=",".join(str(t) for t in DEFAULT_THRESHOLDS))
    ap.add_argument("--per_table", action="store_true", help="print per-document detail")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        run_selfcheck()
        return
    if not (args.gt_dir and args.pred_one_dir and args.pred_two_dir):
        ap.error("--gt_dir, --pred_one_dir and --pred_two_dir are required (or use --selfcheck)")
    thresholds = [float(x) for x in args.thresholds.split(",")]
    per = compute_per_table(args.gt_dir, args.pred_one_dir, args.pred_two_dir,
                            ignore_file=args.ignore_file, clr_threshold=args.clr_threshold)
    print_report(per, thresholds, args.clr_threshold, per_table_detail=args.per_table)


if __name__ == "__main__":
    main()
