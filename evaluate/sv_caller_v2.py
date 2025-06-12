#!/usr/bin/env python3

import os
import re
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import pysam # Required for BAM/FASTA reading

# Optional, but recommended for efficient interval-based clustering if implemented
# from intervaltree import Interval, IntervalTree

# ==============================================================================
# Section 1: CIGAR Preprocessing (User-Provided Function)
# ==============================================================================

def modify_cigar_for_svs(cigar_string, read_length, match_threshold=25):
    """
    Original user-provided function.
    Merges SV operations (D/I/N) separated by small matches (< threshold),
    and potentially adds artificial soft clips based on read_length.

    Args:
        cigar_string (str): The input CIGAR string.
        read_length (int): The length of the read sequence.
        match_threshold (int): Max length of 'M' ops to merge across.

    Returns:
        tuple: (merged_cigar_string, soft_clipped_cigar_string)
               We primarily use the first element (merged_cigar_string).
    """
    # --- Part 1: Merging SVs separated by small Matches ---
    cigar_ops = re.findall(r'(\d+)([MIDNSHP=X])', cigar_string)
    if not cigar_ops:
        return cigar_string, cigar_string # Return original if empty/invalid

    cigar_ops = [(int(length), op) for length, op in cigar_ops]

    new_cigar_list = []
    i = 0
    while i < len(cigar_ops):
        length, op = cigar_ops[i]

        if op in "DIN": # Start of a potential SV group
            group = [(length, op)]
            j = i + 1
            while j < len(cigar_ops):
                next_len, next_op = cigar_ops[j]
                # Check if next is small match or another SV op
                is_small_match = (next_op == 'M' and next_len < match_threshold)
                is_sv_op = (next_op in "DIN")

                if is_small_match or is_sv_op:
                    group.append((next_len, next_op))
                    j += 1
                else:
                    break # End of group

            # Process the group
            sv_lengths = Counter()
            total_group_len = 0
            has_sv = False
            for l, o in group:
                total_group_len += l
                if o in "DIN":
                    sv_lengths[o] += l
                    has_sv = True

            if has_sv: # If the group contained at least one D/I/N
                # Decide dominant SV op based on total length
                dominant_op = sv_lengths.most_common(1)[0][0]
                # Sum lengths of ALL ops in the group (D,I,N and small M) for the new op length
                new_cigar_list.append((total_group_len, dominant_op))
            else:
                 # Should not happen if group started with D/I/N, but as fallback:
                 new_cigar_list.extend(group)

            i = j # Continue after the processed group
        else: # Not a D/I/N op, just add it
            new_cigar_list.append((length, op))
            i += 1

    merged_cigar_string = ''.join(f"{l}{o}" for l, o in new_cigar_list)

    # --- Part 2: Artificial Soft Clipping (from original code - use with caution) ---
    soft_clip_cigar_list = list(new_cigar_list) # Operate on a copy
    try: # Wrap in try-except as this logic might be brittle
        temp_cigar_ops_for_sc = [(str(l), o) for l,o in soft_clip_cigar_list]

        first_match_len = 0
        last_match_len = 0
        for k, (l_str, o) in enumerate(temp_cigar_ops_for_sc):
            if o == 'M':
                first_match_len = int(l_str)
                break
        for k, (l_str, o) in reversed(list(enumerate(temp_cigar_ops_for_sc))):
             if o == 'M':
                 last_match_len = int(l_str)
                 break

        reverse_cigar = (first_match_len < last_match_len) # Heuristic from original
        if reverse_cigar:
            temp_cigar_ops_for_sc.reverse()

        processed_len_count = 0
        final_ops = []
        stop_processing = False
        for k, (l_str, op) in enumerate(temp_cigar_ops_for_sc):
            if stop_processing: break
            length = int(l_str)

            read_consuming_len = 0
            if op in 'MIS=X':
                read_consuming_len = length

            if processed_len_count + read_consuming_len <= read_length:
                 final_ops.append((l_str, op))
                 processed_len_count += read_consuming_len
            else: # Exceeds read length
                 remaining_len = read_length - processed_len_count
                 if remaining_len > 0 and op in 'MIS=X':
                     final_ops.append((str(remaining_len), 'S'))
                 stop_processing = True
                 processed_len_count = read_length

        if reverse_cigar:
             final_ops.reverse()

        soft_clip_cigar = ''.join(f"{l}{o}" for l, o in final_ops)

    except Exception:
        soft_clip_cigar = merged_cigar_string

    return merged_cigar_string, soft_clip_cigar


# ==============================================================================
# Section 2: Core SV Signal Detection (Single Read + Depth Context)
# ==============================================================================

def detect_sv_from_cigar_coverage_v2(
    cigar_string,
    start_pos,
    depth_dict,
    read_length,
    mapping_quality,
    # --- Add a debug flag ---
    debug_read_id=None, # Pass read ID for targeted debugging prints
    # --- Parameters ---
    padding=50,
    min_sv_size=50,
    min_mapping_quality=20,
    del_std_dev_factor=2.0,
    ins_std_dev_factor=1.5,
    clip_breakpoint_ratio_threshold=0.5,
    min_local_depth_for_analysis=5
):
    """
    Improved analysis of CIGAR string and coverage depth for potential SV signals.
    Includes optional debug prints.
    """
    # --- Basic Filters ---
    if not cigar_string or not depth_dict:
        return []

    # --- DEBUG PRINT ---
    is_debug_target = debug_read_id is not None
    # if is_debug_target: print(f"\n--- DEBUG: Analyzing Read {debug_read_id} ---")

    if mapping_quality is not None and mapping_quality != 255 and mapping_quality < min_mapping_quality:
        # if is_debug_target: print(f"  DEBUG: Skipping read - MQ {mapping_quality} < {min_mapping_quality}")
        return []
    effective_mq = mapping_quality if mapping_quality is not None and mapping_quality != 255 else 0

    # --- Step 1: Preprocess CIGAR ---
    try:
        merged_cigar_string, _ = modify_cigar_for_svs(cigar_string, read_length)
        # if is_debug_target and merged_cigar_string != cigar_string:
        #     print(f"  DEBUG: Original CIGAR: {cigar_string}")
        #     print(f"  DEBUG: Merged CIGAR:   {merged_cigar_string}")
    except Exception as e:
        # if is_debug_target: print(f"  DEBUG: modify_cigar_for_svs failed: {e}. Using original CIGAR.")
        merged_cigar_string = cigar_string

    cigar_ops = re.findall(r'(\d+)([MIDNSHP=X])', merged_cigar_string)
    if not cigar_ops: return []

    detected_svs = []
    current_ref_pos = start_pos
    alignment_ref_span = 0

    # --- Step 2: First Pass - Calculate details ---
    op_details = []
    read_pos_consumed = 0
    for i, (length_str, op) in enumerate(cigar_ops):
        length = int(length_str)
        ref_consumed = length if op in ['M', 'D', 'N', '=', 'X'] else 0
        query_consumed = length if op in ['M', 'I', 'S', '=', 'X'] else 0
        op_start_ref_pos = current_ref_pos
        op_start_read_pos = read_pos_consumed
        detail = { 'op': op, 'len': length, 'idx': i, 'ref_start': op_start_ref_pos, 'ref_end': op_start_ref_pos + ref_consumed, 'read_start': op_start_read_pos, 'read_end': op_start_read_pos + query_consumed, 'is_sv_candidate': (op in 'DINS' and length >= min_sv_size) }
        op_details.append(detail)
        current_ref_pos += ref_consumed
        read_pos_consumed += query_consumed
        alignment_ref_span += ref_consumed
    alignment_end_ref_pos = start_pos + alignment_ref_span

    # --- Step 3: Second Pass - Analyze coverage ---
    for detail in op_details:
        if not detail['is_sv_candidate']: continue

        op = detail['op']
        length = detail['len']
        op_ref_start = detail['ref_start']
        op_ref_end = detail['ref_end']

        # if is_debug_target: print(f"  DEBUG: Checking Candidate: {op}{length} at ref {op_ref_start}")

        # --- Determine Context Window ---
        context_start, context_end = -1, -1
        if op in ['D', 'N']:
             context_start = max(1, op_ref_start - padding)
             context_end = op_ref_end + padding
        elif op == 'I':
             insertion_point_ref = op_ref_start
             context_start = max(1, insertion_point_ref - padding)
             context_end = insertion_point_ref + 1 + padding
        elif op == 'S':
            is_trailing_clip = (detail['idx'] == len(op_details) - 1)
            is_leading_clip = (detail['idx'] == 0)
            breakpoint_pos = -1
            if is_leading_clip: breakpoint_pos = start_pos
            elif is_trailing_clip: breakpoint_pos = alignment_end_ref_pos
            if breakpoint_pos != -1 and breakpoint_pos > 0:
                context_start = max(1, breakpoint_pos - padding)
                context_end = breakpoint_pos + padding
            else:
                # if is_debug_target: print(f"    DEBUG: Skipping - Internal soft clip or invalid breakpoint ({alignment_end_ref_pos})")
                continue
        if context_start == -1 or context_end <= context_start:
             # if is_debug_target: print(f"    DEBUG: Skipping - Invalid context window {context_start}-{context_end}")
             continue

        # --- Calculate Local Context Stats ---
        context_positions = sorted([pos for pos in depth_dict if context_start <= pos < context_end])
        if not context_positions:
            # if is_debug_target: print(f"    DEBUG: Skipping - No depth positions found in context {context_start}-{context_end}")
            continue
        context_depths = [depth_dict[pos] for pos in context_positions]
        if not context_depths: continue

        local_avg_depth = np.mean(context_depths)
        local_std_depth = np.std(context_depths)

        # if is_debug_target:
        #     print(f"    DEBUG: Context {context_start}-{context_end}: AvgDepth={local_avg_depth:.2f}, StdDev={local_std_depth:.2f}")

        if local_avg_depth < min_local_depth_for_analysis:
            # if is_debug_target: print(f"    DEBUG: Skipping - Local avg depth {local_avg_depth:.2f} < {min_local_depth_for_analysis}")
            continue

        # --- Deletion / Skipped Region Analysis ---
        if op in ['D', 'N']:
            sv_type = 'deletion' if op == 'D' else 'skipped_region'
            sv_positions = [pos for pos in context_positions if op_ref_start <= pos < op_ref_end]
            sv_avg_depth = np.mean([depth_dict[pos] for pos in sv_positions]) if sv_positions else 0
            threshold_depth = local_avg_depth - (del_std_dev_factor * local_std_depth)
            depth_ratio = sv_avg_depth / local_avg_depth if local_avg_depth > 1e-6 else 0
            passes_check = sv_avg_depth < threshold_depth

            # if is_debug_target:
            #     print(f"    DEBUG: DEL/SKIP Check: SV_AvgDepth={sv_avg_depth:.2f}, Threshold={threshold_depth:.2f}, Ratio={depth_ratio:.3f}")
            #     print(f"    DEBUG: DEL/SKIP Check Result: {'PASS' if passes_check else 'FAIL'}")

            if passes_check:
                detected_svs.append({ "start": op_ref_start, "end": op_ref_end - 1, "length": length, "type": sv_type, "local_avg_depth": round(local_avg_depth, 2), "local_std_depth": round(local_std_depth, 2), "sv_avg_depth": round(sv_avg_depth, 2), "depth_ratio": round(depth_ratio, 3), "cigar_op": f"{length}{op}", "read_mq": effective_mq })

        # --- Insertion Analysis ---
        elif op == 'I':
            sv_type = 'insertion'
            insertion_point_ref = op_ref_start
            left_positions = [pos for pos in context_positions if pos < insertion_point_ref]
            right_positions = [pos for pos in context_positions if pos >= insertion_point_ref]
            if not left_positions or not right_positions:
                # if is_debug_target: print(f"    DEBUG: Skipping INS - Missing flanking context at {insertion_point_ref}")
                continue
            left_avg = np.mean([depth_dict[pos] for pos in left_positions])
            right_avg = np.mean([depth_dict[pos] for pos in right_positions])
            flank_abs_diff = abs(left_avg - right_avg)
            threshold_diff = ins_std_dev_factor * max(1.0, local_std_depth)
            flank_diff_ratio = flank_abs_diff / local_avg_depth if local_avg_depth > 1e-6 else 0
            passes_check = flank_abs_diff < threshold_diff

            # if is_debug_target:
            #     print(f"    DEBUG: INS Check: LeftAvg={left_avg:.2f}, RightAvg={right_avg:.2f}, AbsDiff={flank_abs_diff:.2f}, ThresholdDiff={threshold_diff:.2f}")
            #     print(f"    DEBUG: INS Check Result: {'PASS' if passes_check else 'FAIL'}")

            if passes_check:
                 detected_svs.append({ "pos": insertion_point_ref, "length": length, "type": sv_type, "local_avg_depth": round(local_avg_depth, 2), "local_std_depth": round(local_std_depth, 2), "left_avg_depth": round(left_avg, 2), "right_avg_depth": round(right_avg, 2), "flank_abs_diff": round(flank_abs_diff, 2), "flank_diff_ratio": round(flank_diff_ratio, 3), "cigar_op": f"{length}{op}", "read_mq": effective_mq })

        # --- Soft Clip Analysis ---
        elif op == 'S':
            is_trailing_clip = (detail['idx'] == len(op_details) - 1)
            is_leading_clip = (detail['idx'] == 0)
            breakpoint_pos = -1
            clip_side = None
            if is_leading_clip:
                breakpoint_pos = start_pos
                clip_side = 'start'
            elif is_trailing_clip:
                breakpoint_pos = alignment_end_ref_pos
                clip_side = 'end'

            if breakpoint_pos != -1 and breakpoint_pos > 0:
                 sv_type = 'soft_clip_breakpoint'
                 left_positions = [pos for pos in context_positions if pos < breakpoint_pos]
                 right_positions = [pos for pos in context_positions if pos >= breakpoint_pos]
                 if not left_positions and not right_positions:
                    # if is_debug_target: print(f"    DEBUG: Skipping CLIP - No flanking depth at breakpoint {breakpoint_pos}")
                    continue
                 left_avg = np.mean([depth_dict.get(pos, 0) for pos in left_positions]) if left_positions else 0
                 right_avg = np.mean([depth_dict.get(pos, 0) for pos in right_positions]) if right_positions else 0
                 max_flank_avg = max(left_avg, right_avg, 1e-6)
                 min_flank_avg = min(left_avg, right_avg)
                 breakpoint_depth_ratio = min_flank_avg / max_flank_avg
                 is_significant_drop = breakpoint_depth_ratio < clip_breakpoint_ratio_threshold
                 passes_check = is_significant_drop # Check threshold

                #  if is_debug_target:
                #      print(f"    DEBUG: CLIP Check ({clip_side}): Breakpoint={breakpoint_pos}, LeftAvg={left_avg:.2f}, RightAvg={right_avg:.2f}, Ratio={breakpoint_depth_ratio:.3f}")
                #      print(f"    DEBUG: CLIP Check Result (Ratio < {clip_breakpoint_ratio_threshold}): {'PASS' if passes_check else 'FAIL'}")

                 if passes_check:
                     detected_svs.append({ "breakpoint_pos": breakpoint_pos, "clip_length": length, "clip_side": clip_side, "type": sv_type, "local_avg_depth": round(local_avg_depth, 2), "local_std_depth": round(local_std_depth, 2), "left_avg_depth": round(left_avg, 2), "right_avg_depth": round(right_avg, 2), "breakpoint_depth_ratio": round(breakpoint_depth_ratio, 3), "is_significant_drop": is_significant_drop, "cigar_op": f"{length}{op}", "read_mq": effective_mq })

    # if is_debug_target and detected_svs:
    #     print(f"  DEBUG: Detected {len(detected_svs)} SV signals for read {debug_read_id}")
    # elif is_debug_target:
    #     print(f"  DEBUG: No SV signals detected for read {debug_read_id}")

    return detected_svs


# ==============================================================================
# Section 3: Clustering and Filtering Functions
# ==============================================================================

def cluster_svs_by_type(potential_svs, pos_window=100, length_ratio_similarity=0.3):
    """
    Clusters potential SVs based on genomic proximity, type, and length similarity.
    (Logic remains the same as the previous version)
    """
    if not potential_svs:
        return []
    sv_by_type = defaultdict(list)
    for sv in potential_svs:
        sv_type = sv['type']
        if sv_type == 'soft_clip_breakpoint':
            sv_key = f"{sv_type}_{sv.get('clip_side', 'unknown')}"
        elif sv_type in ['deletion', 'skipped_region']:
             sv_key = 'deletion_skip'
        else:
             sv_key = sv_type
        sv_by_type[sv_key].append(sv)

    final_clusters = []
    for sv_key, sv_list in sv_by_type.items():
        if not sv_list: continue
        sv_list.sort(key=lambda x: x.get('start', x.get('pos', x.get('breakpoint_pos', 0))))
        used_indices = set()
        for i in range(len(sv_list)):
            if i in used_indices: continue
            sv1 = sv_list[i]
            pos1_start = sv1.get('start', sv1.get('pos', sv1.get('breakpoint_pos', 0)))
            len1 = sv1.get('length', sv1.get('clip_length', 0))
            current_cluster_data = {
                'type': sv1['type'], 'cluster_key': sv_key, 'supporting_svs': [sv1],
                'read_ids': {sv1['read_id']}, 'mq_sum': sv1['read_mq'],
                'starts': [pos1_start], 'ends': [sv1.get('end', pos1_start)], 'lengths': [len1],
            }
            if sv1['type'] == 'soft_clip_breakpoint':
                current_cluster_data['clip_side'] = sv1.get('clip_side', 'unknown')
            used_indices.add(i)
            for j in range(i + 1, len(sv_list)):
                if j in used_indices: continue
                sv2 = sv_list[j]
                pos2_start = sv2.get('start', sv2.get('pos', sv2.get('breakpoint_pos', 0)))
                len2 = sv2.get('length', sv2.get('clip_length', 0))
                start_close = abs(pos1_start - pos2_start) <= pos_window
                end_close = True
                if sv_key == 'deletion_skip':
                    pos1_end = sv1.get('end', pos1_start + len1 -1)
                    pos2_end = sv2.get('end', pos2_start + len2 -1)
                    end_close = abs(pos1_end - pos2_end) <= pos_window
                length_similar = True
                if len1 > 0 and len2 > 0:
                    max_len = max(len1, len2)
                    if max_len < 10: length_similar = (abs(len1 - len2) / max_len) < 0.5
                    else: length_similar = (abs(len1 - len2) / max_len) < length_ratio_similarity
                if start_close and end_close and length_similar:
                    if sv2['read_id'] not in current_cluster_data['read_ids']:
                        current_cluster_data['supporting_svs'].append(sv2)
                        current_cluster_data['read_ids'].add(sv2['read_id'])
                        current_cluster_data['mq_sum'] += sv2['read_mq']
                        current_cluster_data['starts'].append(pos2_start)
                        current_cluster_data['ends'].append(sv2.get('end', pos2_start))
                        current_cluster_data['lengths'].append(len2)
                        used_indices.add(j)
            support_count = len(current_cluster_data['read_ids'])
            avg_mq = current_cluster_data['mq_sum'] / len(current_cluster_data['supporting_svs']) if len(current_cluster_data['supporting_svs']) > 0 else 0
            try:
                median_start = int(np.median(current_cluster_data['starts']))
                median_end = int(np.median(current_cluster_data['ends']))
                median_length = int(np.median(current_cluster_data['lengths']))
                start_range_val = (min(current_cluster_data['starts']), max(current_cluster_data['starts']))
                end_range_val = (min(current_cluster_data['ends']), max(current_cluster_data['ends']))
                length_range_val = (min(current_cluster_data['lengths']), max(current_cluster_data['lengths']))
            except ValueError:
                median_start, median_end, median_length = 0, 0, 0
                start_range_val, end_range_val, length_range_val = (0,0), (0,0), (0,0)
            finalized_cluster = {
                'type': current_cluster_data['type'], 'support_count': support_count, 'avg_mq': avg_mq,
                'median_start': median_start, 'median_end': median_end, 'median_length': median_length,
                'start_range': start_range_val, 'end_range': end_range_val, 'length_range': length_range_val,
                'supporting_svs': current_cluster_data['supporting_svs'], 'cluster_key': sv_key,
                **({'clip_side': current_cluster_data['clip_side']} if 'clip_side' in current_cluster_data else {})
            }
            final_clusters.append(finalized_cluster)
    return final_clusters


def score_and_filter_sv_clusters(
    clusters,
    min_support=3,
    min_avg_mq=30,
    max_start_precision=50 # Max allowed spread of start positions
    ):
    """
    Scores clusters and filters based on support, quality, and precision.
    (Includes precision filter and scoring adjustments)
    """
    filtered_clusters = []
    for cluster in clusters:
        # --- Basic Filters ---
        if cluster['support_count'] < min_support:
            continue
        if cluster['avg_mq'] < min_avg_mq:
            continue

        # --- Precision Filter ---
        start_precision = cluster['start_range'][1] - cluster['start_range'][0]
        cluster['start_precision'] = start_precision # Store precision
        if start_precision > max_start_precision:
             continue # Filter if start positions are too spread out

        # --- Calculate Score ---
        normalized_mq = min(cluster['avg_mq'], 60.0) / 60.0
        base_score = cluster['support_count'] * normalized_mq

        # --- Scoring Adjustments (Boost for precision) ---
        precision_boost = 0.0
        if start_precision <= 10: precision_boost = base_score * 0.2
        elif start_precision <= 25: precision_boost = base_score * 0.1

        # Calculate length consistency
        min_len, max_len = cluster['length_range']
        # Avoid division by zero for zero-length events (like some insertions?)
        length_consistency_ratio = (max_len - min_len) / max(1.0, float(max_len)) if max_len > 0 else 0.0
        cluster['length_consistency_ratio'] = round(length_consistency_ratio, 3)
        length_penalty = 0.0
        if length_consistency_ratio > 0.5: length_penalty = base_score * 0.1

        cluster['score'] = base_score + precision_boost - length_penalty
        filtered_clusters.append(cluster)

    # Sort by score descending
    filtered_clusters.sort(key=lambda x: x['score'], reverse=True)
    return filtered_clusters


# ==============================================================================
# Section 4: CIGAR Visualization Helper
# ==============================================================================

def visualize_cigar(cigar_string, start_pos, ax, y_pos=0.5, color_map=None, height=0.6):
    """
    Draws CIGAR operations on a matplotlib axis. Improved version.
    (Logic remains the same as the previous version)
    """
    if not cigar_string: return
    if color_map is None:
        color_map = { 'M': '#87CEEB', '=': '#4682B4', 'X': '#FF6347', 'I': '#90EE90', 'D': '#CD5C5C', 'N': '#D2B48C', 'S': '#FFA500', 'H': '#808080', 'P': '#D3D3D3' }
    try: cigar_ops = re.findall(r'(\d+)([MIDNSHP=X])', cigar_string)
    except Exception as e: ax.text(0.5, y_pos, f"Invalid CIGAR: {cigar_string[:30]}...", color='red', ha='center', va='center'); return
    current_ref_pos = float(start_pos); alignment_started = False; total_prefix_len = 0
    temp_ref_pos = float(start_pos)
    for length_str, op in cigar_ops:
        length = int(length_str)
        if op in ['S', 'H', 'I'] and not alignment_started: total_prefix_len += length
        elif op in ['M', '=', 'X', 'D', 'N']: alignment_started = True
    viz_prefix_start_pos = float(start_pos) - total_prefix_len
    current_ref_pos = float(start_pos); current_viz_pos = float(start_pos); prefix_drawn_offset = 0.0; suffix_drawn_offset = 0.0; alignment_started = False; last_ref_consuming_end_pos = float(start_pos)
    min_coord_drawn = float('inf'); max_coord_drawn = float('-inf')
    for i, (length_str, op) in enumerate(cigar_ops):
        length = int(length_str); color = color_map.get(op, 'gray'); alpha = 0.75; edge_color = 'darkslategray'; linewidth = 0.5; op_height = height * 0.8
        viz_start = 0; viz_width = 0
        if op in ['M', '=', 'X', 'D', 'N']:
            alignment_started = True; viz_start = current_ref_pos; viz_width = float(length); current_ref_pos += viz_width; last_ref_consuming_end_pos = current_ref_pos
            linestyle = '-'; hatch = None
            if op == 'D': op_height = height * 0.3; linewidth = 0
            elif op == 'N': op_height = height * 0.2; linestyle = '--'; linewidth = 1.0; ax.plot([viz_start, viz_start + viz_width], [y_pos, y_pos], color=color, linestyle=linestyle, linewidth=linewidth)
            elif op == 'X': hatch = 'x'
            if op not in ['N']:
                 rect = mpatches.Rectangle((viz_start, y_pos - op_height / 2), viz_width, op_height,linewidth=linewidth, edgecolor=edge_color, facecolor=color, alpha=alpha,linestyle=linestyle, hatch=hatch); ax.add_patch(rect)
            if viz_width > 5: ax.text(viz_start + viz_width / 2, y_pos, op, ha='center', va='center', fontsize=7, color='white' if op not in ['N'] else 'black', fontweight='bold')
            min_coord_drawn = min(min_coord_drawn, viz_start); max_coord_drawn = max(max_coord_drawn, viz_start + viz_width)
        elif op == 'I':
            op_height = height * 0.7; ins_viz_width = max(1.0, length * 0.05); ins_viz_width = min(ins_viz_width, 4.0)
            if alignment_started:
                 viz_start = current_ref_pos - ins_viz_width / 2
                 rect = mpatches.Rectangle((viz_start, y_pos + height * 0.05), ins_viz_width, op_height, linewidth=linewidth, edgecolor=edge_color, facecolor=color, alpha=alpha); ax.add_patch(rect)
                 ax.text(current_ref_pos, y_pos + height * 0.05 + op_height, f"I{length}", ha='center', va='bottom', fontsize=6, color='black')
            else:
                 viz_start = viz_prefix_start_pos + prefix_drawn_offset; viz_width = ins_viz_width; prefix_drawn_offset += viz_width
                 rect = mpatches.Rectangle((viz_start, y_pos - op_height / 2), viz_width, op_height, linewidth=linewidth, edgecolor=edge_color, facecolor=color, alpha=alpha); ax.add_patch(rect)
                 ax.text(viz_start + viz_width / 2, y_pos, f"I{length}", ha='center', va='center', fontsize=6, color='black')
                 min_coord_drawn = min(min_coord_drawn, viz_start); max_coord_drawn = max(max_coord_drawn, viz_start + viz_width)
        elif op in ['S', 'H']:
            op_height = height * 0.75; clip_viz_width = max(1.5, length * 0.08); clip_viz_width = min(clip_viz_width, 6.0)
            if alignment_started:
                 viz_start = last_ref_consuming_end_pos + suffix_drawn_offset; viz_width = clip_viz_width; suffix_drawn_offset += viz_width
                 rect = mpatches.Rectangle((viz_start, y_pos - op_height / 2), viz_width, op_height, linewidth=linewidth, edgecolor=edge_color, facecolor=color, alpha=alpha, hatch='/' if op == 'H' else None); ax.add_patch(rect)
                 ax.text(viz_start + viz_width / 2, y_pos, f"{op}{length}", ha='center', va='center', fontsize=6, color='black'); max_coord_drawn = max(max_coord_drawn, viz_start + viz_width)
            else:
                 viz_start = viz_prefix_start_pos + prefix_drawn_offset; viz_width = clip_viz_width; prefix_drawn_offset += viz_width
                 rect = mpatches.Rectangle((viz_start, y_pos - op_height / 2), viz_width, op_height, linewidth=linewidth, edgecolor=edge_color, facecolor=color, alpha=alpha, hatch='/' if op == 'H' else None); ax.add_patch(rect)
                 ax.text(viz_start + viz_width / 2, y_pos, f"{op}{length}", ha='center', va='center', fontsize=6, color='black')
                 min_coord_drawn = min(min_coord_drawn, viz_start); max_coord_drawn = max(max_coord_drawn, viz_start + viz_width)
        elif op == 'P': pass
    label_x_pos = min(min_coord_drawn, float(start_pos) - total_prefix_len) - 30
    ax.set_yticks([])


# ==============================================================================
# Section 5: Main Aggregated SV Calling Workflow
# ==============================================================================

def call_svs_using_depth_graphs_aggregated(
    reads_near_alignments,
    to_be_sv_called_reads, # DataFrame containing primary reads info
    reference_path, # Pass path to the reference FASTA file
    output_folder,
    # --- Parameters for SV detection and clustering ---
    enable_debug_prints=False,
    padding=50,
    min_sv_size=50,
    min_mapping_quality=20,
    del_std_dev_factor=2.0,
    ins_std_dev_factor=1.5,
    clip_breakpoint_ratio_threshold=0.5,
    min_local_depth_for_analysis=5,
    cluster_pos_window=100,
    cluster_len_similarity=0.4,
    # --- Adjusted Defaults for Higher Confidence ---
    min_read_support=4,         # Increased from 3
    min_cluster_avg_mq=35,      # Increased from 30
    max_start_precision=50,     # Added precision filter
    plot_enabled=True
):
    """
    Calls structural variants (SVs) by aggregating CIGAR-based evidence.
    Includes updated filtering and reporting for higher confidence.
    """
    # --- Setup ---
    if plot_enabled:
        plt.rcParams['figure.dpi'] = 120; plt.rcParams['figure.figsize'] = (14, 10); plt.rcParams['font.size'] = 9
        plt.rcParams['axes.linewidth'] = 1.0; plt.rcParams['axes.grid'] = True; plt.rcParams['grid.alpha'] = 0.4
        plt.rcParams['grid.linestyle'] = ':'; plt.style.use('seaborn-v0_8-whitegrid')
    sv_results = []
    plot_dir = os.path.join(output_folder, "sv_plots_aggregated")
    if plot_enabled: os.makedirs(plot_dir, exist_ok=True)

    print(f"Initializing reference access: {reference_path}")
    try: ref_fasta = pysam.FastaFile(reference_path); print("Reference FASTA opened successfully.")
    except Exception as e: raise RuntimeError(f"Failed to open reference FASTA file {reference_path}: {e}")

    print(f"Starting SV analysis for {len(to_be_sv_called_reads)} primary regions...")
    processed_count = 0
    for idx, row in to_be_sv_called_reads.iterrows():
        processed_count += 1
        read_id_enum = row["enum"]; primary_read_id = row.get("read_id", f"enum_{read_id_enum}"); primary_chrom = row.get("chrom", "19")
        target_debug_read_enum = None; do_debug = enable_debug_prints or (target_debug_read_enum is not None and read_id_enum == target_debug_read_enum)
        if do_debug: print(f"\n=== DEBUG MODE ENABLED FOR REGION {read_id_enum} ===")
        print(f"\n[{processed_count}/{len(to_be_sv_called_reads)}] Processing region for primary read: {primary_read_id} (Enum: {read_id_enum}, Chrom: {primary_chrom})")

        # --- Input Checks ---
        if primary_chrom is None: print(f"  Skipping: Missing 'chrom'."); continue
        if primary_chrom not in ref_fasta.references: print(f"  Skipping: Chrom '{primary_chrom}' not in reference."); continue
        if read_id_enum not in reads_near_alignments: print(f"  Skipping: No nearby reads found."); continue
        nearby_reads = reads_near_alignments[read_id_enum]
        if not nearby_reads: print(f"  Skipping: Empty nearby reads list."); continue

        # --- Define Analysis Region ---
        try:
            main_start_pos = int(row["read/frag_gt_idx"]); main_cigar = row["alignment_cigar_str"]
            # Calculate reference span robustly using re.findall (corrected)
            main_alignment_len_on_ref = 0
            ops = re.findall(r'(\d+)([MDN=X])', main_cigar) # Ops consuming reference
            for length_str, op in ops:
                main_alignment_len_on_ref += int(length_str)
            if main_alignment_len_on_ref <= 0: main_alignment_len_on_ref = row.get("alignment_length", 100) # Fallback
            main_end_pos = main_start_pos + main_alignment_len_on_ref
        except Exception as e: print(f"  Warning: Could not parse start/CIGAR: {e}. Skipping region."); continue
        buffer = 300; chrom_len = ref_fasta.get_reference_length(primary_chrom)
        analysis_start = max(1, main_start_pos - buffer); analysis_end = min(chrom_len + 1, main_end_pos + buffer)
        if analysis_end <= analysis_start: print(f"  Skipping: Invalid analysis range ({analysis_start}-{analysis_end})."); continue


        # --- Calculate Depth ---
        depth_dict = defaultdict(int)
        print(f"  Calculating depth for region {primary_chrom}:{analysis_start}-{analysis_end-1} using {len(nearby_reads)} reads...")
        max_calculated_depth = 0
        for read in nearby_reads:
             if read.is_unmapped or read.reference_start is None or read.reference_name != primary_chrom: continue
             read_start_0based = read.reference_start; read_end_0based = read.reference_end
             if read_end_0based is None or read_end_0based <= analysis_start -1 or read_start_0based >= analysis_end -1: continue
             try:
                 for pos_0based in read.get_reference_positions(full_length=False):
                      pos_1based = pos_0based + 1
                      if analysis_start <= pos_1based < analysis_end:
                          depth_dict[pos_1based] += 1; max_calculated_depth = max(max_calculated_depth, depth_dict[pos_1based])
             except Exception: pass
        if not depth_dict: print(f"  Skipping: No depth calculated."); continue
        positions = sorted(depth_dict.keys())
        if not positions: print(f"  Skipping: No positions with depth."); continue
        depths = [depth_dict[pos] for pos in positions]; avg_depth = np.mean(depths) if depths else 0; std_depth = np.std(depths) if depths else 0
        print(f"  Region {primary_chrom}:{analysis_start}-{analysis_end-1}: Avg Depth={avg_depth:.2f}, StdDev={std_depth:.2f}, Max Depth={max_calculated_depth}")
        if do_debug: print(f"  DEBUG: Depth Dict (first 10): {list(depth_dict.items())[:10]}")

        # --- Collect Potential SVs ---
        all_potential_svs = []
        print(f"  Detecting potential SV signals in {len(nearby_reads)} nearby reads...")
        for read in nearby_reads:
            if read.is_unmapped or read.reference_start is None or read.reference_name != primary_chrom: continue
            nearby_cigar = read.cigarstring; nearby_start_pos = read.reference_start; nearby_read_len = read.query_length; nearby_mq = read.mapping_quality; read_name = read.query_name
            if not nearby_cigar or nearby_start_pos is None or nearby_mq is None: continue
            read_end_0based = read.reference_end
            if read_end_0based is None or read_end_0based <= analysis_start -1 or nearby_start_pos >= analysis_end -1: continue
            debug_id_to_pass = read_name if do_debug else None
            potential_svs_from_read = detect_sv_from_cigar_coverage_v2(
                nearby_cigar, nearby_start_pos + 1, depth_dict, nearby_read_len, nearby_mq,
                debug_read_id=debug_id_to_pass, padding=padding, min_sv_size=min_sv_size, min_mapping_quality=min_mapping_quality,
                del_std_dev_factor=del_std_dev_factor, ins_std_dev_factor=ins_std_dev_factor,
                clip_breakpoint_ratio_threshold=clip_breakpoint_ratio_threshold, min_local_depth_for_analysis=min_local_depth_for_analysis
            )
            for sv in potential_svs_from_read:
                sv_start = sv.get('start', sv.get('pos', sv.get('breakpoint_pos', -1))); sv_end = sv.get('end', sv_start)
                if sv_start != -1 and max(analysis_start, sv_start) < min(analysis_end, sv_end + 1):
                    sv['read_id'] = read_name; all_potential_svs.append(sv)
        print(f"  Found {len(all_potential_svs)} potential SV signals initially within region.")
        if do_debug and all_potential_svs: print(f"  DEBUG: Potential SVs (first 5): {all_potential_svs[:5]}")
        if not all_potential_svs: print(f"  No potential SV signals found meeting criteria."); # continue

        # --- Cluster SVs ---
        print(f"  Clustering {len(all_potential_svs)} potential SV signals...")
        sv_clusters = cluster_svs_by_type(all_potential_svs, pos_window=cluster_pos_window, length_ratio_similarity=cluster_len_similarity)
        print(f"  Generated {len(sv_clusters)} initial clusters.")

        # --- Score and Filter Clusters (Using updated function and parameters) ---
        print(f"  Scoring and filtering clusters (MinSupport={min_read_support}, MinAvgMQ={min_cluster_avg_mq}, MaxStartPrecision={max_start_precision})...")
        filtered_sv_clusters = score_and_filter_sv_clusters(
            sv_clusters,
            min_support=min_read_support,
            min_avg_mq=min_cluster_avg_mq,
            max_start_precision=max_start_precision # Apply precision filter
        )
        print(f"  Found {len(filtered_sv_clusters)} high-confidence SV clusters passing filters.")
        if do_debug and filtered_sv_clusters: print(f"  DEBUG: Filtered Clusters (first 3): {filtered_sv_clusters[:3]}")

        # --- Plotting (if enabled) ---
        plot_path = None
        if plot_enabled:
            print(f"  Generating plot...")
            # *** UPDATED gridspec for 4 plots ***
            fig = plt.figure(figsize=(14, 12)) # Increased height slightly
            gs = gridspec.GridSpec(4, 1, height_ratios=[4, 1, 1, 1], hspace=0.15) # 4 rows

            # --- Depth Plot (ax1) ---
            ax1 = plt.subplot(gs[0])
            ax1.plot(positions, depths, '-', color='steelblue', linewidth=1.0, alpha=0.9, label='Read Depth')
            ax1.axhline(y=avg_depth, color='darkgreen', linestyle='--', linewidth=1.5, label=f'Avg Depth: {avg_depth:.2f}')
            ax1.fill_between(positions, 0, depths, color='skyblue', alpha=0.3)
            color_map = { "deletion": "red", "skipped_region": "brown", "deletion_skip": "red", "insertion": "darkviolet", "soft_clip_breakpoint_start": "darkorange", "soft_clip_breakpoint_end": "goldenrod", "soft_clip_breakpoint_unknown": "gray" }
            sorted_clusters = sorted(filtered_sv_clusters, key=lambda x: x['median_start'])
            label_y_coords = np.linspace(max(depths) * 0.95 if depths else 10, max(max(depths)*0.5, avg_depth * 1.5) if depths else 5, max(len(sorted_clusters),1) + 1)
            label_y_idx = 0
            for i, cluster in enumerate(sorted_clusters):
                sv_id = i + 1; sv_type = cluster['type']; start = cluster['median_start']; end = cluster['median_end'] if sv_type in ['deletion', 'skipped_region'] else start; length = cluster['median_length']; support = cluster['support_count']; avg_mq_clus = cluster['avg_mq']
                color_key = cluster.get('cluster_key', sv_type);
                if sv_type == 'soft_clip_breakpoint': color_key = f"{sv_type}_{cluster.get('clip_side', 'unknown')}"
                color = color_map.get(color_key, "dimgray")
                if sv_type in ['deletion', 'skipped_region']:
                     ax1.axvspan(start, end + 1, alpha=0.35, color=color, zorder=1, label=f"_{sv_type}")
                     label_anchor_x = (start + end) / 2.0; label_text = f"({sv_id}) {sv_type[:3].upper()} {length}bp\nSupp:{support} MQ:{avg_mq_clus:.0f}"
                elif sv_type == 'insertion':
                     ax1.axvline(x=start, color=color, linestyle='-', linewidth=3, alpha=0.6, zorder=2, label="_insertion")
                     label_anchor_x = start; label_text = f"({sv_id}) INS {length}bp\nSupp:{support} MQ:{avg_mq_clus:.0f}"
                elif sv_type == 'soft_clip_breakpoint':
                     ax1.axvline(x=start, color=color, linestyle='--', linewidth=3, alpha=0.6, zorder=2, label="_soft_clip")
                     side = cluster.get('clip_side', '?').upper(); label_anchor_x = start; label_text = f"({sv_id}) CLIP ({side}) {length}bp\nSupp:{support} MQ:{avg_mq_clus:.0f}"
                else:
                     ax1.axvspan(start, end + 1, alpha=0.35, color=color, zorder=1, label="_other_sv")
                     label_anchor_x = (start + end) / 2.0; label_text = f"({sv_id}) {sv_type[:3].upper()} {length}bp\nSupp:{support} MQ:{avg_mq_clus:.0f}"
                y_pos_label = label_y_coords[label_y_idx % len(label_y_coords)]; label_y_idx += 1
                ax1.text(label_anchor_x, y_pos_label, label_text, ha='center', va='center', fontsize=7, bbox=dict(facecolor='white', alpha=0.85, edgecolor=color, boxstyle='round,pad=0.2'), zorder=5)
            ax1.axvline(x=main_start_pos, color='darkcyan', linestyle=':', linewidth=1.5, alpha=0.8, label='Primary Align Start')
            ax1.axvline(x=main_end_pos, color='darkred', linestyle=':', linewidth=1.5, alpha=0.8, label='Primary Align End (Approx)')
            ax1.set_ylabel('Read Depth', fontsize=11); title_str = f'Aggregated SV Analysis: {primary_chrom}:{analysis_start}-{analysis_end-1}\nRegion near {primary_read_id} (Enum: {read_id_enum})'; ax1.set_title(title_str, fontsize=13)
            ax1.grid(True, alpha=0.4, linestyle=':'); ax1.legend(loc='upper right', fontsize=8); ax1.set_xlim(analysis_start - 10, analysis_end + 10);
            # *** Hide x-ticks for top plot ***
            ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

            # --- Main Aligned CIGAR Plot (ax2) ---
            ax2 = plt.subplot(gs[1], sharex=ax1) # Share x-axis
            main_cigar = row["alignment_cigar_str"]
            try:
                temp_cigar_ops = re.findall(r'(\d+)([MIDNSHP=X])', main_cigar); viz_start_pos = main_start_pos
                if temp_cigar_ops and temp_cigar_ops[0][1] == 'S': ref_consumed = sum(int(l) for l, op in temp_cigar_ops if op in 'MDN=X'); viz_start_pos = main_end_pos - ref_consumed
                visualize_cigar(main_cigar, viz_start_pos, ax2, y_pos=0.5)
            except Exception as e: ax2.text(0.5, 0.5, f"Error visualizing: {main_cigar}", ha='center', va='center', color='red', fontsize=8)
            ax2.set_yticks([]); ax2.set_ylabel('Primary CIGAR', fontsize=10);
            # *** Hide x-ticks for middle plot ***
            ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

            # --- Modified CIGAR Plot (ax3) ---
            ax3 = plt.subplot(gs[2], sharex=ax1) # Share x-axis
            # *** Get modified CIGAR (Assuming it was calculated and stored) ***
            # This assumes you run modify_cigar_for_svs and store its first output
            # For demonstration, let's recalculate it here (inefficient but works)
            try:
                primary_read_seq_len = len(row.get("read_gt", "")) # Need read length
                if primary_read_seq_len > 0:
                     modified_cigar, _ = modify_cigar_for_svs(main_cigar, primary_read_seq_len)
                else:
                     modified_cigar = main_cigar # Fallback if length unknown
            except Exception as e:
                 print(f"    Warning: Failed to run modify_cigar_for_svs: {e}")
                 modified_cigar = main_cigar # Fallback

            try:
                temp_cigar_ops_mod = re.findall(r'(\d+)([MIDNSHP=X])', modified_cigar); viz_start_pos_mod = main_start_pos
                if temp_cigar_ops_mod and temp_cigar_ops_mod[0][1] == 'S': ref_consumed_mod = sum(int(l) for l, op in temp_cigar_ops_mod if op in 'MDN=X'); viz_start_pos_mod = main_end_pos - ref_consumed_mod
                visualize_cigar(modified_cigar, viz_start_pos_mod, ax3, y_pos=0.5)
            except Exception as e:
                 print(f"    Warning: Failed to visualize modified CIGAR {modified_cigar}: {e}")
                 ax3.text(0.5, 0.5, f"Error visualizing Mod: {modified_cigar}", ha='center', va='center', color='red', fontsize=8)
            ax3.set_yticks([]); ax3.set_ylabel('Modified CIGAR', fontsize=10);
            # *** Hide x-ticks for middle plot ***
            ax3.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)


            # --- Ground Truth CIGAR Plot (ax4) ---
            # *** Use gs[3] for the 4th subplot ***
            ax4 = plt.subplot(gs[3], sharex=ax1) # Share x-axis
            gt_cigar = row["read_gt_cigar_str"]
            try:
                temp_cigar_ops_gt = re.findall(r'(\d+)([MIDNSHP=X])', gt_cigar); viz_start_pos_gt = main_start_pos
                if temp_cigar_ops_gt and temp_cigar_ops_gt[0][1] == 'S': ref_consumed_gt = sum(int(l) for l, op in temp_cigar_ops_gt if op in 'MDN=X'); viz_start_pos_gt = main_end_pos - ref_consumed_gt
                visualize_cigar(gt_cigar, viz_start_pos_gt, ax4, y_pos=0.5)
            except Exception as e: ax4.text(0.5, 0.5, f"Error visualizing GT: {gt_cigar}", ha='center', va='center', color='red', fontsize=8)
            ax4.set_yticks([]); ax4.set_ylabel('GT CIGAR', fontsize=10);
            # *** Show x-label only on the bottom plot ***
            ax4.set_xlabel(f'Reference Position ({primary_chrom})', fontsize=11)

            # --- Add CIGAR Legend to bottom plot (ax4) ---
            cigar_color_map_legend = {'M': '#87CEEB','=': '#4682B4','X': '#FF6347','I': '#90EE90','D': '#CD5C5C','N': '#D2B48C','S': '#FFA500','H': '#808080','P': '#D3D3D3'}
            # Include ops from all 3 CIGARs shown in the legend
            used_ops_in_plot = set(op for _, op in re.findall(r'(\d+)([MIDNSHP=X])', main_cigar + modified_cigar + gt_cigar))
            legend_handles = [mpatches.Patch(color=cigar_color_map_legend.get(op, 'gray'), label=op) for op in sorted(list(used_ops_in_plot)) if op in cigar_color_map_legend]
            if legend_handles: ax4.legend(handles=legend_handles, loc='lower right', title='CIGAR Ops', ncol=len(legend_handles), fontsize=7)

            # --- Finalize and Save Plot ---
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plot_path = os.path.join(plot_dir, f"aggregated_sv_plot_{read_id_enum}.png")
            try: plt.savefig(plot_path, dpi=150); print(f"  Plot saved to: {plot_path}")
            except Exception as e: print(f"  Error saving plot {plot_path}: {e}"); plot_path = None
            plt.close(fig)

        # --- Step 5: Reporting Results ---
        # (Reporting logic remains the same, using filtered_sv_clusters)
        for cluster in filtered_sv_clusters:
            sorted_support_svs = sorted(cluster['supporting_svs'], key=lambda x: x['read_mq'], reverse=True)
            best_sv_rep = sorted_support_svs[0] if sorted_support_svs else {}
            sv_result = {
                "primary_read_enum": read_id_enum, "primary_read_id": primary_read_id, "chrom": primary_chrom,
                "reference_region": f"{analysis_start}-{analysis_end-1}",
                "sv_type": cluster['type'], "sv_start": cluster['median_start'], "sv_end": cluster['median_end'],
                "sv_length": cluster['median_length'], "support_reads": cluster['support_count'], "avg_mq": round(cluster['avg_mq'], 1),
                "cluster_score": round(cluster['score'], 2), "start_precision": cluster['start_precision'],
                "length_consistency_ratio": cluster['length_consistency_ratio'],
                "length_range_min": cluster['length_range'][0], "length_range_max": cluster['length_range'][1],
                "rep_local_avg_depth": round(best_sv_rep.get("local_avg_depth", np.nan), 2),
                "rep_depth_ratio": round(best_sv_rep.get("depth_ratio", np.nan), 3),
                "rep_flank_diff_ratio": round(best_sv_rep.get("flank_diff_ratio", np.nan), 3),
                "rep_breakpoint_depth_ratio": round(best_sv_rep.get("breakpoint_depth_ratio", np.nan), 3),
                "rep_cigar_op": best_sv_rep.get("cigar_op", ""),
                "plot_path": plot_path if plot_enabled else "Plotting disabled",
                **({'clip_side': cluster['clip_side']} if 'clip_side' in cluster else {})
            }
            sv_results.append(sv_result)

    # --- Final Output ---
    print("\nConsolidating final results...")
    sv_results_df = pd.DataFrame(sv_results)
    print(f"Generated DataFrame with {len(sv_results_df)} aggregated SV calls.")
    ref_fasta.close()
    return sv_results_df

def sv_results_refiner(results_df):
    eval_results = results_df
    new_column_names = ["enum",
                    "read_gt", 
                    "read_gt_idx", 
                    "read_gt_cigar_str", 
                    "read/frag_gt_reference_interval", 
                    "is_read_gt_index_exist_in_top75", 
                    "read/frag_gt_top75_index", 
                    "is_read/frag_gt_top_match_in_top75", 
                    "read/frag_gt", 
                    "read/frag_gt_idx", 
                    "best_sw_score_in_top75", 
                    "read/frag_gt_best_sw_score",
                    "is_read/frag_gt_sw_score_best_in_top75",
                    "is_read/frag_gt_index_same_as_gt_index",
                    "alignment_str",
                    "second_best_alignment_str",
                    "third_best_alignment_str",
                    "alignment_cigar_str",
                    "second_best_alignment_cigar_str",
                    "third_best_alignment_cigar_str",
                    "modified_alignment_cigar_str",
                    "second_modified_alignment_cigar_str",
                    "third_modified_alignment_cigar_str",
                    "read/frag_gt_alignment_index",
                    "read/frag_gt_alignment_str"]
    
    for idx, name in enumerate(new_column_names):
        eval_results.rename(columns={eval_results.columns[idx]: name}, inplace=True)
        
    eval_results["is_read_gt_index_exist_in_top75"] = eval_results["is_read_gt_index_exist_in_top75"].astype(bool)
    eval_results.loc[eval_results["is_read_gt_index_exist_in_top75"] == False, 
                 [
                 "read/frag_gt_top75_index", 
                 "is_read/frag_gt_top_match_in_top75", 
                 "read/frag_gt", 
                 "read/frag_gt_idx", 
                 "best_sw_score_in_top75", 
                 "read/frag_gt_best_sw_score",
                 "is_read/frag_gt_sw_score_best_in_top75",
                 "is_read/frag_gt_index_same_as_gt_index"]] = None
    
    eval_results["is_read/frag_gt_index_same_as_gt_index"] = None
    eval_results["is_read/frag_gt_index_same_as_gt_index"] = eval_results[eval_results["is_read/frag_gt_sw_score_best_in_top75"] == True].apply(lambda row: abs(row["read_gt_idx"] - row["read/frag_gt_idx"]) <= 10, axis=1)
    eval_results["read_gt_idx_diff_from_read/frag_gt_index"] = eval_results["read_gt_idx"] - eval_results["read/frag_gt_idx"]
    
    return eval_results
