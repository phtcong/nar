# TSR-NAR Demo: Table-Level and Cell-Level Rejection with ConGrid

A minimal, self-contained demo of the rejection pipeline from **TSR-NAR**
(Table Structure Recognition with Novelty and Ambiguity Rejection). It shows how two
independently trained detectors can be combined into a selective predictor that
abstains when they disagree, using the **ConGrid** consensus metric as the agreement
signal.

The demo runs entirely on **pre-computed predictions** (ConGrid JSON). It performs no
model inference, so it needs no GPU, no checkpoints, and no deep-learning stack.

## The pipeline

For each table, two models supply predictions (here RT-DETR-X and ConRTF):

- **TLR (Table-Level Rejection).** Agreement between the two models is measured as the
  ConGrid Cell-F1 of one against the other. Tables whose agreement falls below a
  threshold are rejected, so the system abstains on the whole table.
- **CLR (Cell-Level Rejection).** On accepted tables, only the cells where the two
  models agree are kept. The retained cells are scored against ground truth.
- **ConGrid.** A Consensus- and Content-aware Grid metric that computes element-based
  Precision, Recall, and F1 over rows, columns, and cells, scoring cells by exact text
  match. It is the common currency for both the agreement signal and the final
  evaluation.

The report breaks the two mechanisms out separately, mirroring the paper: Baseline,
CLR only, TLR only, and TLR + CLR across a threshold sweep.

The primary metric is **ERR Δ**, the share of the baseline model's cell errors removed at
a given operating point: ERR Δ = (precision − baseline) / (1 − baseline) × 100, equivalently
the reduction in %ERR (where %ERR = 100 − ConGrid-Cell precision). It is reported on the
CLR-only line and the TLR + CLR sweep. Precision matters more than F1 here, since rejection
raises precision while it can lower recall, so the report shows both per table.

## Install

This project uses [uv](https://docs.astral.sh/uv/). From this directory:

```bash
uv sync                      # 4 runtime deps: numpy, scipy, pandas, pillow
```

Without uv, a plain virtualenv works too:

```bash
pip install -r requirements.txt
```

## Quickstart

Run the bundled self-check (synthetic fixtures, asserts the sweep behaves):

```bash
uv run python demo/nar_demo.py --selfcheck
```

Run the report on the 10 bundled example documents:

```bash
uv run python demo/nar_demo.py \
    --gt_dir       demo/examples/gt \
    --pred_one_dir demo/examples/rtdetr \
    --pred_two_dir demo/examples/conrtf \
    --per_table
```

Regenerate the per-document pipeline figures:

```bash
uv run python demo/nar_viz.py \
    --gt_dir       demo/examples/gt \
    --pred_one_dir demo/examples/rtdetr \
    --pred_two_dir demo/examples/conrtf \
    --images_dir   demo/examples/images \
    --out_dir      demo/examples/viz
```

`--pred_one_dir` is the reference model (RT-DETR-X). `--pred_two_dir` is the primary,
kept model (ConRTF), whose cells survive into the CLR output.

## Layout

```
demo/
  nar_demo.py        report: baseline / CLR / TLR / TLR+CLR sweep
  nar_pipeline.py    TLR, CLR and baseline scoring on ConGrid evaluators
  nar_viz.py         4-panel per-document figure (RT-DETR, ConRTF, TLR, CLR)
  congrid_io.py      loads ConGrid JSON into the evaluator's input shape
  fixtures/          tiny synthetic tables for --selfcheck
  examples/          10 real example documents (see note below)
    gt/ rtdetr/ conrtf/   ConGrid JSON per document
    images/               table crops
    viz/                  rendered pipeline figures
congrid/             the ConGrid evaluator and conversion utilities
```

## Data format

Each ConGrid JSON describes one table:

```json
{
  "width": 600, "height": 400,
  "rows":    [{"bbox": [x1,y1,x2,y2], "text": "...", "score": 0.97}],
  "columns": [{"bbox": [x1,y1,x2,y2], "text": "...", "score": 0.95}],
  "cells":   []
}
```

To run on your own data, produce one such file per table for ground truth and for each
model, then point the three `--*_dir` flags at the matching folders. File basenames must
match across the three directories.

## Note on the examples

The 10 bundled documents come from PubTables-1M* (the PubTables-1M test split with the
351 documents that carry ground truth errors removed). They are a curated subset chosen
to span the range of pipeline behavior: some where the two models agree and nothing is
rejected, some where cell-level rejection trims disagreeing cells, and some low-consensus
tables that table-level rejection would flag. They illustrate how TSR-NAR works and are
not a representative benchmark.

Because the subset deliberately oversamples hard cases, its absolute baseline scores are
lower than PubTables-1M* as a whole and should not be read as model performance. The
meaningful quantity is ERR Δ, the primary metric in the paper: the share of the baseline
model's cell errors removed at a given operating point. ERR Δ appears on the CLR-only line
and the TLR+CLR sweep of the report.

Because ConGrid scores cells by exact text match, ground truth and predictions must share
the same word source for the numbers to be meaningful.

## Scope

This demo covers the rejection and evaluation pipeline only. Model training and
inference, checkpoint loading, and dataset preparation are out of scope. The optional
`convert` extra (`uv sync --extra convert`, adds pymupdf) is needed only if you
regenerate ConGrid JSON from raw model output with the bundled conversion utilities.
