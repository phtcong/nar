import numpy as np
from scipy.optimize import linear_sum_assignment
import json
import os
import pandas as pd
from difflib import SequenceMatcher
# ------------------- ConGrid Class -------------------
class ConGrid:
    def __init__(self,
                weight_distance=0.3,
                weight_iou_threshold=0.3,
                row_type_field = 'section_rows',
                row_iou_threshold = 0.5,
                row_text_threshold = 0.5,
                col_type_field = '',
                col_iou_threshold = 0.5,
                col_text_threshold = 0.5,
                cell_iou_threshold = 0.2,
                cell_text_threshold = 0.6,
                cell_text_threshold_gt = 0.6,
                gt_check=False,
                perfect_text_match_skip=True):
        """
        ConGrid: Consensus- and Content-Aware Grid Evaluation Metric
        for Semantic Table Structure Recognition

        Parameters
        ----------
        weight_iou_threshold : float
            IoU threshold for dynamic weighting
        weight_distance : float
            Maximum weight for center distance contribution
        """
        self.weight_distance        = weight_distance
        self.weight_iou_threshold   = weight_iou_threshold
        self.row_type_field         = row_type_field
        self.row_iou_threshold      = row_iou_threshold
        self.row_text_threshold     = row_text_threshold
        self.col_type_field         = col_type_field
        self.col_iou_threshold      = col_iou_threshold
        self.col_text_threshold     = col_text_threshold
        self.cell_iou_threshold     = cell_iou_threshold
        self.cell_text_threshold    = cell_text_threshold
        self.cell_text_threshold_gt = cell_text_threshold_gt
        
        self.gt_check = gt_check
        self.perfect_text_match_skip = perfect_text_match_skip
        self.row_scores = None
        self.row_semantic_scores = None
        self.col_scores = None
        self.cell_scores = None
        self.data_gt    = None
        self.data_pred  = None
        self.data_pred_one  = None
        self.data_pred_two  = None

    # ------------------- Utility functions -------------------

    def iou(self, bbox1, bbox2):
        ixmin = max(bbox1[0], bbox2[0])
        iymin = max(bbox1[1], bbox2[1])
        ixmax = min(bbox1[2], bbox2[2])
        iymax = min(bbox1[3], bbox2[3])
        if ixmax <= ixmin:
            return 0.0
        if iymax <= iymin:
            return 0.0
        iw = max(ixmax - ixmin, 0.)
        ih = max(iymax - iymin, 0.)
        inter = iw * ih
        union = (bbox1[2]-bbox1[0])*(bbox1[3]-bbox1[1]) + (bbox2[2]-bbox2[0])*(bbox2[3]-bbox2[1]) - inter
        return inter / union if union > 0 else 0.0

    def f_iou(self, iou):
        """Dynamic weight for center distance"""
        if iou < self.weight_iou_threshold:
            return 0
        return self.weight_distance

    def center_distance_score(self, bbox1, bbox2, axis='row'):
        if axis == 'row':
            d = abs((bbox1[1] + bbox1[3]) - (bbox2[1] + bbox2[3])) / 2.0
            m = max(bbox1[3], bbox1[3]) - min(bbox1[1], bbox1[1])
            return max(0.0, 1 - d/m)
        else:
            d = abs((bbox1[0] + bbox1[2]) - (bbox2[0] + bbox2[2])) / 2.0
            m = max(bbox1[2], bbox1[2]) - min(bbox1[0], bbox1[0])
            return max(0.0, 1 - d/m)
    def get_element_types(self, type_field, data):
        if type_field != '':
            if type_field in data:
                return data[type_field]
        return None
    def get_row_types(self, data):
        return self.get_element_types(self.row_type_field, data)
    def get_column_types(self, data):
        return self.get_element_types(self.col_type_field, data)
    def match_rows(self):
        pred_types = None
        gt_types = None
        if self.row_type_field != '':
            if self.row_type_field in self.data_pred:
                pred_types = self.data_pred[self.row_type_field]
                gt_types = self.data_gt[self.row_type_field]
            
        return self.match_entities(
                    self.data_pred['rows'], 
                    self.data_gt['rows'], 
                    axis='row', 
                    pred_types = pred_types, 
                    gt_types= gt_types
              )
    
    def match_columns(self):
        pred_types = None
        gt_types = None
        if self.col_type_field != '':
            if self.col_type_field in self.data_pred:
                pred_types = self.data_pred[self.col_type_field]
                gt_types = self.data_gt[self.col_type_field]
            
        return self.match_entities(
                    self.data_pred['columns'], 
                    self.data_gt['columns'], 
                    axis='col', 
                    pred_types = pred_types, 
                    gt_types= gt_types
              )
    # ------------------- Hungarian matching -------------------
    def match_entities(self, pred, gt, axis='row', pred_types=None, gt_types=None):
        n_pred = len(pred)
        n_gt = len(gt)
        if n_pred == 0 or n_gt == 0:
            return {}, [0.0]*n_pred, [0.0]*n_pred, [0.0]*n_pred

        score_matrix = np.zeros((n_pred, n_gt))
        iou_matrix = np.zeros((n_pred, n_gt))
        D_matrix = np.zeros((n_pred, n_gt))
        for i in range(n_pred):
            for j in range(n_gt):
                if pred_types is not None:
                    if pred_types[i] != gt_types[j]:
                        iou_ij = 0
                        D = 1.0
                        score_matrix[i, j] = 0
                        iou_matrix[i, j] = iou_ij
                        D_matrix[i, j] = D
                        continue
                iou_ij = self.iou(pred[i], gt[j])
                D = self.center_distance_score(pred[i], gt[j], axis=axis)
                score_matrix[i, j] = self.compute_entity_score(iou_ij, D)
                iou_matrix[i, j] = iou_ij
                D_matrix[i, j] = D

        cost_matrix = 1 - score_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_indices = {i: j for i, j in zip(row_ind, col_ind)}
        S_scores = [score_matrix[i, matched_indices[i]] if i in matched_indices else 0.0
                    for i in range(n_pred)]
        S_ious = [iou_matrix[i, matched_indices[i]] if i in matched_indices else 0.0
                    for i in range(n_pred)]
        S_distances = [D_matrix[i, matched_indices[i]] if i in matched_indices else 0.0
                    for i in range(n_pred)]
        return matched_indices, S_scores, S_ious, S_distances

    # ------------------- Score formulas -------------------
    def compute_entity_score(self, iou, D):
        if iou == 0:
            return 0
        w = self.f_iou(iou)
        return (1 - w) * iou + w * D

    def compute_cell_score_formula(self, cell_iou, S_text):
        if S_text == 1.0:
            return 1.0
        if cell_iou < self.cell_iou_threshold:
            return 0.0
        if not self.gt_check:
            if S_text < self.cell_text_threshold:
                return 0
            return S_text
        if S_text < self.cell_text_threshold_gt:
            return 0
        return S_text    
        
    # ------------------- Cell-level scoring -------------------
    def compute_cell_scores(self, pred_cells, gt_cells, row_mapping, col_mapping, text_sim_fn):
        S_cells = []
        cell_mapping = []
        map_gt_cells = {}
        map_gt_section_cells = {}
        
        for idx, c in enumerate(gt_cells):
            c['idx'] = idx
            if c['section']:
                key = f"{c['row']}"
            else:
                key = f"{c['row']}-{c['col']}"
            map_gt_cells[key] = c
        for cell in pred_cells:
            r_pred, c_pred = cell['row'], cell['col']
            if r_pred not in row_mapping :
                S_cells.append(0.0)
                cell_mapping.append(None)
                continue
            
            #check secion cell
            if cell['section']:
                r_gt = row_mapping[r_pred]
                key = f"{r_gt}"
                if key not in map_gt_cells:
                    S_cells.append(0.0)
                    cell_mapping.append(None)
                    continue
                gt_cell = map_gt_cells[key]
                gt_idx = gt_cell['idx'] 
            else:
                #check normal cell
                if c_pred not in col_mapping:
                    S_cells.append(0.0)
                    cell_mapping.append(None)
                    continue
                r_gt, c_gt = row_mapping[r_pred], col_mapping[c_pred]
                key = f"{r_gt}-{c_gt}"
                if key not in map_gt_cells:
                    S_cells.append(0.0)
                    cell_mapping.append(None)
                    continue
                gt_cell = map_gt_cells[key]
                gt_idx = gt_cell['idx'] 
            
            cell_iou = self.iou(cell['bbox'], gt_cell['bbox'])
            S_text = text_sim_fn(cell['text'], gt_cell['text'])
            cell_score = self.compute_cell_score_formula(cell_iou, S_text)            
            S_cells.append(cell_score)
            if (cell_score > 0): 
                cell_mapping.append(gt_idx)
            else:
                cell_mapping.append(None)
        return np.array(S_cells), cell_mapping

    # ------------------- Aggregation -------------------
    def compute_mean_scores(self, scores, n=0):
        if n > 0:
            return np.sum(scores) / n
        if len(scores) > 0:
            return np.mean(scores)
        return 0.0
    def compute_row_semantic_scores(self, pred_text_rows, gt_text_rows, row_scores, row_mapping, text_sim_fn):
        ret = []
        if len(row_scores) < 1:
            return ret
        
        for r_pred, score in enumerate(row_scores):
            semantic_score = 0
            if r_pred in row_mapping:
                S_text_row = text_sim_fn(pred_text_rows[r_pred], gt_text_rows[row_mapping[r_pred]])                                      
                semantic_score = self.compute_row_semantic_score_formula(score, S_text_row)
            ret.append(semantic_score)
        return ret
    def compute_row_semantic_score_formula(self, score, S_text):
        if self.perfect_text_match_skip and S_text == 1.0:
            return 1.0
        if score < self.row_iou_threshold:
            return 0.0

        if not self.gt_check:
            return S_text

        if S_text < self.row_text_threshold:
            return 0
        return S_text

    def compute_col_semantic_scores(self, pred_text_cols, gt_text_cols, col_scores, col_mapping, text_sim_fn):
        ret = []
        if len(col_scores) < 1:
            return ret

        for c_pred, score in enumerate(col_scores):
            semantic_score = 0
            if c_pred in col_mapping:
                S_text_col = text_sim_fn(pred_text_cols[c_pred], gt_text_cols[col_mapping[c_pred]])
                semantic_score = self.compute_col_semantic_score_formula(score, S_text_col)
            ret.append(semantic_score)
        return ret

    def compute_col_semantic_score_formula(self, score, S_text):
        if self.perfect_text_match_skip and S_text == 1.0:
            return 1.0
        if score < self.col_iou_threshold:
            return 0.0

        if not self.gt_check:
            return S_text

        if S_text < self.col_text_threshold:
            return 0
        return S_text
    
    def text_sim(self, pred_text, gt_text):
        if pred_text == gt_text:
            return 1.0
        return 0
    def text_sim_one_line(self, pred_text, gt_text):
        if pred_text == gt_text:
            return 1.0
        a = "" if pd.isna(pred_text) else str(pred_text)
        b = "" if pd.isna(gt_text) else str(gt_text)
        if not a and not b:
            return 0.0

        m = SequenceMatcher(None, a, b)
        match = m.find_longest_match(0, len(a), 0, len(b))
        return match.size / max(len(a), len(b))
    def text_sim_multi_lines(self, pred_text, gt_text):
        matcher = SequenceMatcher(None, pred_text, gt_text)        
        matching_blocks = matcher.get_matching_blocks()
        return matcher.ratio()
    def cell_text_sim(self, pred_text, gt_text):
        return self.text_sim(pred_text, gt_text)
    def row_text_sim(self, pred_text, gt_text):
        return self.text_sim_one_line(pred_text, gt_text)
    def col_text_sim(self, pred_text, gt_text):
        return self.text_sim_multi_lines(pred_text, gt_text)
    # ------------------- Public API -------------------
    def setData(self, data_gt, data_pred):
        self.data_gt    = data_gt
        self.data_pred  = data_pred
    def setDataCLR(self, cell_data_gt, cell_data_pred_one, cell_data_pred_two):
        self.cell_data_gt    = cell_data_gt
        self.cell_data_pred_one  = cell_data_pred_one
        self.cell_data_pred_two  = cell_data_pred_two
    def f1_score(self, precision, recall):
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)    
    def evaluate(self, 
                    cell_text_sim_fn=None, 
                    row_text_sim_fn=None,
                    col_text_sim_fn=None,
                ):
        if cell_text_sim_fn is None:
            cell_text_sim_fn = self.cell_text_sim
        if row_text_sim_fn is None:
            row_text_sim_fn = self.row_text_sim
        if col_text_sim_fn is None:
            col_text_sim_fn = self.col_text_sim
        
        self.row_mapping, self.row_scores, self.row_ious, self.row_distances = self.match_rows()
        self.col_mapping, self.col_scores, self.col_ious, self.col_distances = self.match_columns()

        self.cell_scores, self.cell_mapping = self.compute_cell_scores(self.data_pred['cells'], self.data_gt['cells'], self.row_mapping, self.col_mapping, cell_text_sim_fn)
        self.row_semantic_scores = self.compute_row_semantic_scores(self.data_pred['text_rows'], self.data_gt['text_rows'], self.row_scores, self.row_mapping, row_text_sim_fn)
        self.col_semantic_scores = self.compute_col_semantic_scores(self.data_pred.get('text_columns', []), self.data_gt.get('text_columns', []), self.col_scores, self.col_mapping, col_text_sim_fn)
        
        row_semantic_precision  = self.compute_mean_scores(self.row_semantic_scores, len(self.data_pred['rows']))
        row_semantic_recall     = self.compute_mean_scores(self.row_semantic_scores, len(self.data_gt['rows']))
        row_semantic_f1 = self.f1_score(row_semantic_precision, row_semantic_recall)

        col_semantic_precision  = self.compute_mean_scores(self.col_semantic_scores, len(self.data_pred['columns']))
        col_semantic_recall     = self.compute_mean_scores(self.col_semantic_scores, len(self.data_gt['columns']))
        col_semantic_f1 = self.f1_score(col_semantic_precision, col_semantic_recall)

        cell_precision  = self.compute_mean_scores(self.cell_scores, len(self.data_pred['cells']))
        cell_recall     = self.compute_mean_scores(self.cell_scores, len(self.data_gt['cells']))
        cell_f1 = self.f1_score(cell_precision, cell_recall)

        row_struct_precision  = self.compute_mean_scores(self.row_scores, len(self.data_pred['rows']))
        row_struct_recall     = self.compute_mean_scores(self.row_scores, len(self.data_gt['rows']))
        row_struct_f1 = self.f1_score(row_struct_precision, row_struct_recall)

        col_struct_precision  = self.compute_mean_scores(self.col_scores, len(self.data_pred['columns']))
        col_struct_recall     = self.compute_mean_scores(self.col_scores, len(self.data_gt['columns']))
        col_struct_f1 = self.f1_score(col_struct_precision, col_struct_recall)
        
        return {
            'cell_precision': cell_precision,
            'cell_recall': cell_recall,
            'cell_f1': cell_f1,

            'row_semantic_precision': row_semantic_precision,
            'row_semantic_recall': row_semantic_recall,
            'row_semantic_f1': row_semantic_f1,

            'col_semantic_precision': col_semantic_precision,
            'col_semantic_recall': col_semantic_recall,
            'col_semantic_f1': col_semantic_f1,

            'row_struct_precision': row_struct_precision,
            'row_struct_recall': row_struct_recall,
            'row_struct_f1': row_struct_f1,

            'col_struct_precision': col_struct_precision,
            'col_struct_recall': col_struct_recall,
            'col_struct_f1': col_struct_f1,
            
            
            
            'gt_num_row': len(self.data_gt['rows']),
            'gt_num_col': len(self.data_gt['columns']),
            'gt_num_cell': len(self.data_gt['cells']),
            
            'pred_num_row': len(self.data_pred['rows']),
            'pred_num_col': len(self.data_pred['columns']),
            'pred_num_cell': len(self.data_pred['cells']),
            
            'row_mapping': self.row_mapping,
            'col_mapping': self.col_mapping,
            'cell_mapping': self.cell_mapping,
            'row_struct_scores': self.row_scores,
            'row_semantic_scores': self.row_semantic_scores,
            'col_semantic_scores': self.col_semantic_scores,
            'column_struct_scores': self.col_scores,
            'cell_scores': self.cell_scores,
            
        }
    def evaluate_default(self):
        
        return {
            'cell_precision': 0,
            'cell_recall': 0,
            'cell_f1': 0,

            'row_semantic_precision': 0,
            'row_semantic_recall': 0,
            'row_semantic_f1': 0,

            'col_semantic_precision': 0,
            'col_semantic_recall': 0,
            'col_semantic_f1': 0,

            'row_struct_precision': 0,
            'row_struct_recall': 0,
            'row_struct_f1': 0,

            'col_struct_precision': 0,
            'col_struct_recall': 0,
            'col_struct_f1': 0,
            
            
            
            'gt_num_row': 0,
            'gt_num_col': 0,
            'gt_num_cell': 0,
            
            'pred_num_row': 0,
            'pred_num_col': 0,
            'pred_num_cell': 0,
            
            'row_mapping': 0,
            'col_mapping': 0,
            'cell_mapping': 0,
            'row_struct_scores': 0,
            'row_semantic_scores': 0,
            'col_semantic_scores': 0,
            'column_struct_scores': 0,
            'cell_scores': 0,
            
        }
    def evaluate_clr(self, threshold = 1.0):
        self.data_gt    = self.cell_data_pred_one
        self.data_pred  = self.cell_data_pred_two
        matching = self.evaluate()
        cells = self.data_pred['cells']
        
        ifilter = matching['cell_scores'] >= threshold
        
        cell_text_sim_fn = self.cell_text_sim
            
        self.data_gt    = self.cell_data_gt
        
        self.row_mapping, self.row_scores, self.row_ious, self.row_distances = self.match_rows()
        self.col_mapping, self.col_scores, self.col_ious, self.col_distances = self.match_columns()
        pred_cells = [cells[i] for i in range(len(ifilter)) if ifilter[i]]
        
        self.cell_scores, self.cell_mapping = self.compute_cell_scores(pred_cells, self.data_gt['cells'], self.row_mapping, self.col_mapping, cell_text_sim_fn)
        
        cell_precision  = self.compute_mean_scores(self.cell_scores, len(pred_cells))
        cell_recall     = self.compute_mean_scores(self.cell_scores, len(self.data_gt['cells']))
        cell_f1 = self.f1_score(cell_precision, cell_recall)
        
        return {
            'cell_precision': cell_precision,
            'cell_recall': cell_recall,
            'cell_f1': cell_f1,
            'cell_mapping': self.cell_mapping,
            'cell_scores': self.cell_scores,
            'clr_cells': pred_cells,
        }