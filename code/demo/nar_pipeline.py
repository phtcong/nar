import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "congrid", "utils"))
from con_grid import ConGrid  # noqa: E402

sys.path.insert(0, HERE)
from congrid_io import load_congrid_data  # noqa: E402,F401


def make_evaluator():
    """Construct a ConGrid evaluator matching the paper pipeline."""
    return ConGrid(gt_check=1, weight_distance=0.3, weight_iou_threshold=0.3)


def tlr_similarity(data_pred_one, data_pred_two):
    """Table-level agreement between two models = ConGrid Cell-F1 of one vs two."""
    ev = make_evaluator()
    ev.setData(data_pred_one, data_pred_two)  # gt-slot = pred_one
    return ev.evaluate()["cell_f1"]


def clr_score(data_gt, data_pred_one, data_pred_two, clr_threshold=1.0):
    """Cell-Level Rejection consensus, then score retained cells vs GT.

    pred_two is the kept/primary model (ConRTF, f_2 in the paper); pred_one is
    the consensus reference (RT-DETR, f_1). Each pred_two cell is scored against
    pred_one and kept only on agreement, matching the paper's CLR definition.

    Returns the evaluate_clr dict plus cell retention (kept / total pred_two
    cells), so the caller can report cell-level coverage for CLR.
    """
    ev = make_evaluator()
    ev.setDataCLR(data_gt, data_pred_one, data_pred_two)
    out = ev.evaluate_clr(threshold=clr_threshold)
    n_total = len(data_pred_two["cells"])
    n_kept = len(out.get("clr_cells", []))
    out["cell_retention"] = (n_kept / n_total) if n_total else 0.0
    return out


def baseline_score(data_gt, data_pred):
    """ConGrid of a single model vs GT (no rejection), for reference."""
    ev = make_evaluator()
    ev.setData(data_gt, data_pred)
    s = ev.evaluate()
    return {"cell_precision": s["cell_precision"],
            "cell_recall": s["cell_recall"],
            "cell_f1": s["cell_f1"]}
