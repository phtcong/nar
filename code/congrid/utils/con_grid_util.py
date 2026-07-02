from .table import apply_score_threshold, iob, align_table_elements
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from postprocess import nms, sort_objects_left_to_right, sort_objects_top_to_bottom
def objects_to_table_elements(objects, thrh=0.0):
    column_headers, columns, rows = [], [], []
    
    for obj in objects:
        obj['section'] = 0
        if obj['class'] == 0:
            column_headers.append(obj)
        elif obj['class'] == 1:
            columns.append(obj)
        elif obj['class'] == 2:
            rows.append(obj)
        else:
            obj['section'] = 1
            rows.append(obj)
    if thrh > 0:
        column_headers = apply_score_threshold(column_headers, thrh)
        columns = apply_score_threshold(columns, thrh)
        rows = apply_score_threshold(rows, thrh)
        rows = nms(rows, match_criteria="object2_overlap", match_threshold=0.5, keep_higher=True)
        if len(rows) > 1:
            rows = sort_objects_top_to_bottom(rows)
        columns = nms(columns, match_criteria="object2_overlap", match_threshold=0.25, keep_higher=True)
        if len(columns) > 1:
            columns = sort_objects_left_to_right(columns)
        column_headers, columns, rows, _table  = align_table_elements(column_headers, columns, rows)
    return column_headers, columns, rows
def convert_table(headers, rows, cols, words, text_iou_threshold=0.5):
        all_rows = headers + rows
        pred_rows = []
        pred_cols = []
        pred_cells = []
        
        row_idx = 0        
        for row in all_rows:
            row_bbox = row['bbox']
            if row_bbox[2] <= row_bbox[0] or row_bbox[3] <= row_bbox[1]:
                continue
            
            row_words = []            
            row_text = ""
            for word in words:
                check = iob(word['bbox'], row_bbox)
                if (check < text_iou_threshold):
                    continue               
                row_words.append(word['text'])
            row_text = ' '.join(row_words)
            row['text'] = row_text
            row['row'] = row_idx          
            pred_rows.append(row) 
            row_idx += 1       
        col_idx = 0
        for col in cols:
            col_bbox = col['bbox']
            if col_bbox[2] <= col_bbox[0] or col_bbox[3] <= col_bbox[1]:
                continue

            col_words = []
            for word in words:
                check = iob(word['bbox'], col_bbox)
                if check < text_iou_threshold:
                    continue
                col_words.append(word['text'])
            col_text = ' '.join(col_words)
            col['text'] = col_text
            col['col'] = col_idx
            pred_cols.append(col)
            col_idx += 1
            
        row_idx = -1
        for row in pred_rows:
            row_idx += 1
            row_bbox = row['bbox']
            if row['section']:
                pred_cells.append({
                    'section': 1,
                    'bbox': row_bbox,
                    'text': row['text'],
                    'row': row_idx,
                    'col': 0
                })
                continue
            col_idx = -1            
            for col in pred_cols:
                col_idx += 1
                col_bbox = col['bbox']
                cell_bbox = [col_bbox[0],
                             row_bbox[1],
                             col_bbox[2],
                             row_bbox[3]
                            ]
                if cell_bbox[2] <= cell_bbox[0] or cell_bbox[3] <= cell_bbox[1]:
                    continue
                    
                cell_words = []            
                cell_text = ""
                for word in words:            
                    check = iob(word['bbox'], cell_bbox)
                    if (check < text_iou_threshold):
                        continue               
                    cell_words.append(word['text'])
                cell_text = ' '.join(cell_words)
                pred_cells.append({
                    'section': 0,
                    'bbox': cell_bbox,
                    'text': cell_text,
                    'row': row_idx,
                    'col': col_idx
                })
        return pred_rows, pred_cols, pred_cells