import json


def _load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(path, encoding="latin-1") as f:
            return json.load(f)


def _items(objs, field="bbox", default=None):
    out = []
    for o in objs:
        if default is None:
            out.append(o[field])
        else:
            out.append(o.get(field, default))
    return out


def load_congrid_data(path):
    """Load one con_grid JSON into the dict shape ConGrid.evaluate expects.

    Unlike the original sota loader, this also supplies text_columns. Without
    it, ConGrid.evaluate raises IndexError in compute_col_semantic_scores
    whenever predicted columns are matched.
    """
    data = _load_json(path)
    return {
        "rows":          _items(data["rows"]),
        "text_rows":     _items(data["rows"], "text", default=""),
        "row_scores":    _items(data["rows"], "score", default=1.0),
        "columns":       _items(data["columns"]),
        "text_columns":  _items(data["columns"], "text", default=""),
        "column_scores": _items(data["columns"], "score", default=1.0),
        "cells":         data["cells"],
    }
