import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from collections import Counter, defaultdict
from dna2vec.simulate import load_human_reference_genome

def calculate_alignment_integrity(aligned_indices):
    """
    Calculates a structural integrity score for a pairwise alignment based on
    its aligned segment indices.

    The score favors alignments that are less fragmented (fewer chunks)
    and where the aligned portions densely cover the span of the alignment.
    Higher scores indicate better structural integrity. A score of ~2.0
    represents a single, perfect block alignment. Scores decrease as
    fragmentation increases or density decreases.

    Args:
        aligned_indices: A tuple containing two tuples of (start, end) indices
                         for target and query sequences, respectively, as
                         returned by Biopython's alignment.aligned property.
                         Format: (((t_start1, t_end1), ...), ((q_start1, q_end1), ...))

    Returns:
        A float score representing the structural integrity (higher is better),
        or 0.0 if the input is invalid or represents an empty alignment.

    Raises:
        ValueError: If the number of target chunks and query chunks differ.
    """
    # --- Input Validation ---
    if not isinstance(aligned_indices, np.ndarray) or len(aligned_indices) != 2:
        print("Warning: Input must be a tuple of two arrays.")
        return 0.0
    if not isinstance(aligned_indices[0], np.ndarray) or not isinstance(aligned_indices[1], np.ndarray):
         print("Warning: Input must be a tuple of two arrays.")
         return 0.0

    target_chunks = aligned_indices[0]
    query_chunks = aligned_indices[1]

    if len(target_chunks) != len(query_chunks):
        raise ValueError("Target and query chunk lists must have the same length.")

    N = len(target_chunks)
    if N == 0:
        return 0.0 # No alignment chunks, zero integrity

    # --- Calculate Total Aligned Length ---
    # Ensure start <= end and calculate length, summing across chunks
    total_target_aligned = sum(max(0, t_end - t_start) for t_start, t_end in target_chunks)
    total_query_aligned = sum(max(0, q_end - q_start) for q_start, q_end in query_chunks)

    # If total aligned length is zero, integrity is zero
    if total_target_aligned == 0 or total_query_aligned == 0:
        return 0.0

    # --- Calculate Alignment Span ---
    # Span is from the start of the first chunk to the end of the last chunk
    target_span_start = target_chunks[0][0]
    target_span_end = target_chunks[-1][1]
    query_span_start = query_chunks[0][0]
    query_span_end = query_chunks[-1][1]

    # Calculate span length, ensuring it's at least 1 to avoid division by zero
    target_span_length = max(1, target_span_end - target_span_start)
    query_span_length = max(1, query_span_end - query_span_start)

    # --- Calculate Alignment Density ---
    # Density = total aligned length within the span / length of the span
    target_density = total_target_aligned / target_span_length
    query_density = total_query_aligned / query_span_length

    # --- Calculate Final Integrity Score ---
    # Score = Sum of densities penalized by the number of chunks (N)
    # We divide by max(1, N) so N=1 isn't penalized, but N>1 is.
    integrity_score = (target_density + query_density) / max(1, N)

    return integrity_score

def process_alignment_indices(target_seq, query_seq, alignment_indices):
    """Process alignment indices to insert gaps in sequences and create match line."""
    target_indices, query_indices = alignment_indices
    
    # Create arrays to track which positions are aligned
    target_aligned = [False] * len(target_seq)
    query_aligned = [False] * len(query_seq)
    
    # Mark aligned positions
    for i in range(len(target_indices)):
        t_start, t_end = target_indices[i]
        q_start, q_end = query_indices[i]
        
        for j in range(min(t_end - t_start, q_end - q_start)):
            target_aligned[t_start + j] = True
            query_aligned[q_start + j] = True
    
    # Build gapped sequences
    gapped_target = []
    gapped_query = []
    match_line = []
    
    t_idx = q_idx = 0
    
    # Process soft clips at the beginning
    if not query_aligned[0] and len(query_indices) > 0 and query_indices[0][0] > 0:
        for i in range(query_indices[0][0]):
            gapped_target.append('-')
            gapped_query.append(query_seq[i])
            match_line.append('.')
            q_idx = query_indices[0][0]
    
    # Process aligned regions and gaps between them
    for i in range(len(target_indices)):
        t_start, t_end = target_indices[i]
        q_start, q_end = query_indices[i]
        
        # Add any gap between last chunk and this one
        if i > 0:
            last_t_end = target_indices[i-1][1]
            last_q_end = query_indices[i-1][1]
            
            # Add unaligned target sequence (deletion)
            for j in range(last_t_end, t_start):
                gapped_target.append(target_seq[j])
                gapped_query.append('-')
                match_line.append('.')
                
            # Add unaligned query sequence (insertion)
            for j in range(last_q_end, q_start):
                gapped_target.append('-')
                gapped_query.append(query_seq[j])
                match_line.append('.')
        
        # Add the aligned chunk
        for j in range(min(t_end - t_start, q_end - q_start)):
            gapped_target.append(target_seq[t_start + j])
            gapped_query.append(query_seq[q_start + j])
            match_line.append('|' if target_seq[t_start + j] == query_seq[q_start + j] else '.')
    
    # Process soft clips at the end
    if len(query_indices) > 0 and query_indices[-1][1] < len(query_seq):
        for i in range(query_indices[-1][1], len(query_seq)):
            gapped_target.append('-')
            gapped_query.append(query_seq[i])
            match_line.append('.')
    
    return ''.join(gapped_target), ''.join(gapped_query), ''.join(match_line)

def format_alignment(target_seq, query_seq, target_start, cigar_string=None, alignment_indices=None, line_width=60):
    """
    Format the alignment between target and query sequences based on CIGAR string or alignment indices.
    
    Args:
        target_seq: The raw target/reference sequence (without gaps)
        query_seq: The raw query sequence (without gaps)
        target_start: The starting position of the target sequence in the reference
        cigar_string: Optional; the CIGAR string representing the alignment
        alignment_indices: Optional; tuple of target and query alignment indices
        line_width: Width of each line in the alignment display (default: 60)
        
    Returns:
        A formatted string showing the alignment
    """
    # First, insert gaps in the sequences according to alignment information
    if cigar_string is not None:
        gapped_target, gapped_query, match_line = process_cigar(target_seq, query_seq, cigar_string)
    elif alignment_indices is not None:
        gapped_target, gapped_query, match_line = process_alignment_indices(target_seq, query_seq, alignment_indices)
    else:
        raise ValueError("Either cigar_string or alignment_indices must be provided")
    
    # Format the alignment in blocks
    result = []
    target_pos = target_start
    query_pos = 0
    
    for i in range(0, len(gapped_target), line_width):
        block_target = gapped_target[i:i+line_width]
        block_match = match_line[i:i+line_width]
        block_query = gapped_query[i:i+line_width]
        
        # Calculate displayed positions
        target_displayed_chars = sum(1 for c in block_target if c != '-')
        query_displayed_chars = sum(1 for c in block_query if c != '-')
        
        # Use consistent padding for better alignment
        result.append(f"target {target_pos:10d} {block_target}")
        result.append(f"       {i:10d} {block_match}")
        result.append(f"query  {query_pos:10d} {block_query}")
        result.append("")
        
        # Update positions for next block
        target_pos += target_displayed_chars
        query_pos += query_displayed_chars
    
    return "\n".join(result)

def process_cigar(target_seq, query_seq, cigar_string):
    """Process CIGAR string to insert gaps in sequences and create match line."""
    import re
    
    cigar_ops = re.findall(r'(\d+)([MIDNSHP=X])', cigar_string)
    
    gapped_target = []
    gapped_query = []
    match_line = []
    
    t_idx = q_idx = 0
    
    for length, op in cigar_ops:
        length = int(length)
        
        if op == 'M':  # Match or mismatch
            for i in range(length):
                if t_idx < len(target_seq) and q_idx < len(query_seq):
                    gapped_target.append(target_seq[t_idx])
                    gapped_query.append(query_seq[q_idx])
                    match_line.append('|' if target_seq[t_idx] == query_seq[q_idx] else '.')
                    t_idx += 1
                    q_idx += 1
        elif op == 'I':  # Insertion in query
            for i in range(length):
                if q_idx < len(query_seq):
                    gapped_target.append('-')
                    gapped_query.append(query_seq[q_idx])
                    match_line.append('-')
                    q_idx += 1
        elif op == 'D':  # Deletion in query
            for i in range(length):
                if t_idx < len(target_seq):
                    gapped_target.append(target_seq[t_idx])
                    gapped_query.append('-')
                    match_line.append('-')
                    t_idx += 1
        elif op == 'S':  # Soft clipping
            for i in range(length):
                if q_idx < len(query_seq):
                    # Soft clips are in query but not aligned to target
                    gapped_target.append('-')
                    gapped_query.append(query_seq[q_idx])
                    match_line.append(' ')
                    q_idx += 1
    
    return ''.join(gapped_target), ''.join(gapped_query), ''.join(match_line)

def generate_cigar_string(aligned_indices, query_length=None):
    """
    Generate CIGAR string from alignment indices returned by BioPython's PairwiseAlign 
    with Smith-Waterman algorithm, including soft clipping.
    
    Args:
        aligned_indices: A numpy array containing two arrays of start/end indices:
                        [[(t_start1, t_end1), ...], [(q_start1, q_end1), ...]]
                        where t refers to target and q refers to query sequence indices.
        query_length: Optional; the total length of the query sequence for soft clipping.
                     If not provided, soft clips will only be added if clearly needed.
    
    Returns:
        A CIGAR string representing the alignment (e.g., "3S5M2I3M1D7M4S")
    """
    if not isinstance(aligned_indices, np.ndarray) or len(aligned_indices) != 2:
        print("Warning: Input must be an array of two arrays.")
        return "*"
    
    target_indices, query_indices = aligned_indices
    
    if len(target_indices) != len(query_indices):
        raise ValueError("Target and query must have the same number of aligned segments")
    
    if len(target_indices) == 0:
        return "*"  # Return unaligned marker if no alignments
    
    cigar = []
    
    # Add soft clip at the beginning if the alignment doesn't start at the beginning of the query
    if query_indices[0][0] > 0:
        cigar.append(f"{query_indices[0][0]}S")
    
    last_t_end = None
    last_q_end = None
    
    for i in range(len(target_indices)):
        t_start, t_end = target_indices[i]
        q_start, q_end = query_indices[i]
        
        # Handle gaps between aligned segments
        if last_t_end is not None and last_q_end is not None:
            t_gap = t_start - last_t_end
            q_gap = q_start - last_q_end
            
            # Add deletion (gap in query)
            if t_gap > q_gap:
                del_size = t_gap - q_gap
                cigar.append(f"{del_size}D")
            
            # Add insertion (gap in target)
            elif q_gap > t_gap:
                ins_size = q_gap - t_gap
                cigar.append(f"{ins_size}I")
        
        # Add current match
        match_length = min(t_end - t_start, q_end - q_start)
        if match_length > 0:
            cigar.append(f"{match_length}M")
        
        last_t_end = t_end
        last_q_end = q_end
    
    # Add soft clip at the end if the alignment doesn't end at the end of the query
    if query_length is not None and query_indices[-1][1] < query_length:
        soft_clip_end = query_length - query_indices[-1][1]
        if soft_clip_end > 0:
            cigar.append(f"{soft_clip_end}S")
    
    return "".join(cigar)

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

def extract_sv_from_cigar(read, min_sv_size=10):
    """
    Extract potential structural variants from a read's CIGAR string.
    
    Args:
        read: pysam.AlignedSegment object
        min_sv_size: Minimum size to consider as an SV
        
    Returns:
        List of dictionaries containing SV information
    """
    import re
    
    if not read.cigarstring:
        return []
    
    # Parse CIGAR string
    cigar_tuples = read.cigartuples
    if not cigar_tuples:
        return []
    
    svs = []
    ref_pos = read.reference_start
    query_pos = 0
    
    for op, length in cigar_tuples:
        # CIGAR operations: 0=M, 1=I, 2=D, 3=N, 4=S, 5=H, 6=P, 7==, 8=X
        if op == 1 and length >= min_sv_size:  # Insertion
            svs.append({
                'type': 'insertion',
                'start': ref_pos,
                'end': ref_pos,
                'length': length,
                'read_id': read.query_name,
                'query_pos': query_pos
            })
        elif op == 2 and length >= min_sv_size:  # Deletion
            svs.append({
                'type': 'deletion',
                'start': ref_pos,
                'end': ref_pos + length,
                'length': length,
                'read_id': read.query_name,
                'query_pos': query_pos
            })
        elif op == 4 and length >= min_sv_size:  # Soft clip
            if query_pos == 0:  # Start of read
                svs.append({
                    'type': 'soft_clip_start',
                    'start': ref_pos - 1,
                    'end': ref_pos,
                    'length': length,
                    'read_id': read.query_name,
                    'query_pos': query_pos
                })
            else:  # End of read
                svs.append({
                    'type': 'soft_clip_end',
                    'start': ref_pos,
                    'end': ref_pos + 1,
                    'length': length,
                    'read_id': read.query_name,
                    'query_pos': query_pos
                })
        
        # Update positions
        if op in (0, 2, 3, 7, 8):  # Consumes reference
            ref_pos += length
        if op in (0, 1, 4, 7, 8):  # Consumes query
            query_pos += length
    
    return svs

def analyze_aligned_pairs(read, min_sv_size=10):
    """
    Analyze the aligned_pairs of a read to detect structural variants and alignment patterns.
    
    Args:
        read: pysam.AlignedSegment object
        min_sv_size: Minimum size to consider as an SV
        
    Returns:
        Dict with alignment statistics and potential SVs
    """
    # Initialize empty result with default values
    result = {
        'n_matches': 0,
        'n_mismatches': 0,
        'match_percentage': 0,
        'n_deletions': 0,
        'n_insertions': 0,
        'insertions': [],
        'deletions': [],
        'ref_gaps': []
    }
    
    # Skip invalid reads
    if read.is_unmapped or not read.cigartuples:
        return result
    
    try:
        # Get aligned pairs (query_pos, ref_pos)
        aligned_pairs = read.get_aligned_pairs(with_seq=True)
        
        # Initialize counters and tracking variables
        n_matches = 0
        n_mismatches = 0
        n_deletions = 0
        n_insertions = 0
        
        # Track stretches of insertions/deletions to identify SVs
        current_insertion = None
        current_deletion = None
        insertions = []
        deletions = []
        
        # Track reference gaps (possible large deletions)
        ref_positions = []
        query_positions = []
        
        prev_ref_pos = None
        prev_query_pos = None
        
        for query_pos, ref_pos, ref_base in aligned_pairs:
            # Track positions for analysis
            if ref_pos is not None:
                ref_positions.append(ref_pos)
            if query_pos is not None:
                query_positions.append(query_pos)
            
            # Match or mismatch
            if query_pos is not None and ref_pos is not None:
                # Reset any ongoing SV tracking
                if current_insertion:
                    if current_insertion['length'] >= min_sv_size:
                        insertions.append(current_insertion)
                    current_insertion = None
                    
                if current_deletion:
                    if current_deletion['length'] >= min_sv_size:
                        deletions.append(current_deletion)
                    current_deletion = None
                
                # Check for match/mismatch
                if query_pos < len(read.query_sequence):
                    query_base = read.query_sequence[query_pos]
                    if ref_base and ref_base.upper() == query_base.upper():
                        n_matches += 1
                    else:
                        n_mismatches += 1
            
            # Deletion in read (ref base exists, but no query base)
            elif query_pos is None and ref_pos is not None:
                n_deletions += 1
                
                # Start or extend deletion tracking
                if current_deletion is None:
                    current_deletion = {
                        'type': 'deletion',
                        'start': ref_pos,
                        'length': 1,
                        'ref_bases': ref_base if ref_base else ''
                    }
                else:
                    current_deletion['length'] += 1
                    if ref_base:
                        current_deletion['ref_bases'] += ref_base
            
            # Insertion in read (query base exists, but no ref base)
            elif query_pos is not None and ref_pos is None and query_pos < len(read.query_sequence):
                n_insertions += 1
                
                # Start or extend insertion tracking
                if current_insertion is None:
                    current_insertion = {
                        'type': 'insertion',
                        'start': prev_ref_pos + 1 if prev_ref_pos is not None else 0,
                        'length': 1,
                        'query_bases': read.query_sequence[query_pos]
                    }
                else:
                    current_insertion['length'] += 1
                    current_insertion['query_bases'] += read.query_sequence[query_pos]
            
            # Update previous positions
            prev_ref_pos = ref_pos
            prev_query_pos = query_pos
        
        # Handle any ongoing SV at the end
        if current_insertion and current_insertion['length'] >= min_sv_size:
            insertions.append(current_insertion)
        
        if current_deletion and current_deletion['length'] >= min_sv_size:
            deletions.append(current_deletion)
        
        # Check for reference gaps (possible large deletions)
        ref_gaps = []
        if len(ref_positions) > 1:
            for i in range(1, len(ref_positions)):
                gap_size = ref_positions[i] - ref_positions[i-1] - 1
                if gap_size >= min_sv_size:
                    ref_gaps.append({
                        'type': 'ref_gap',
                        'start': ref_positions[i-1],
                        'end': ref_positions[i],
                        'length': gap_size
                    })
        
        # Summary statistics
        total_aligned = n_matches + n_mismatches
        match_pct = (n_matches / total_aligned * 100) if total_aligned > 0 else 0
        
        result = {
            'n_matches': n_matches,
            'n_mismatches': n_mismatches,
            'match_percentage': match_pct,
            'n_deletions': n_deletions,
            'n_insertions': n_insertions,
            'insertions': insertions,
            'deletions': deletions,
            'ref_gaps': ref_gaps
        }
    
    except Exception as e:
        # If there's any error, return the default result with error info
        result['error'] = str(e)
    
    return result

def visualize_cigar(cigar_string, start_pos, ax, y_pos=0.1, color_map=None):
    """
    Visualize a CIGAR string on a matplotlib axis with improved handling
    of leading/trailing operations and insertions.

    Args:
        cigar_string: The CIGAR string to visualize (e.g., "10S90M5I5M").
        start_pos: Starting position on the reference genome/sequence.
        ax: Matplotlib axis object to draw on.
        y_pos: Vertical position for the main alignment track (0-1 range typically).
        color_map: Optional dictionary mapping CIGAR ops ('M', 'I', 'D', etc.) to colors.
    """
    if not cigar_string:
        return

    # Default color map
    if color_map is None:
        color_map = {
            'M': '#87CEEB',  # SkyBlue (Match/Mismatch - often combined)
            '=': '#4682B4',  # SteelBlue (Explicit Match)
            'X': '#FF6347',  # Tomato (Explicit Mismatch)
            'I': '#90EE90',  # LightGreen (Insertion to reference)
            'D': '#CD5C5C',  # IndianRed (Deletion from reference)
            'N': '#D2B48C',  # Tan (Skipped region - e.g., intron)
            'S': '#FFA500',  # Orange (Soft clip)
            'H': '#808080',  # Gray (Hard clip - often not drawn on ref)
            'P': '#D3D3D3',  # LightGray (Padding)
        }

    # Parse CIGAR string
    cigar_ops = re.findall(r'(\d+)([MIDNSHP=X])', cigar_string)
    if not cigar_ops:
        return

    height = 0.03 # Height of the main alignment blocks
    insertion_height = height * 0.8 # Height for insertion markers
    insertion_offset = height * 0.6 # Offset above the main line for insertions
    clip_height = height * 0.9 # Slightly different height for clips? (optional)

    current_ref_pos = float(start_pos) # Use float for potential precision
    current_viz_prefix_pos = float(start_pos) # Tracks the leftmost point for drawing prefixes
    last_ref_consuming_pos = float(start_pos) # Tracks end of last M/D/N/=/X

    processed_ops_details = [] # To store calculated positions before drawing

    # --- Pass 1: Calculate positions ---
    alignment_started = False
    for length_str, op in cigar_ops:
        length = int(length_str)
        op_detail = {'op': op, 'len': length, 'viz_start': 0, 'viz_width': 0}

        if op in ['M', '=', 'X', 'D', 'N']:
            alignment_started = True
            op_detail['viz_start'] = current_ref_pos
            op_detail['viz_width'] = length
            current_ref_pos += length
            last_ref_consuming_pos = current_ref_pos # Update end position
            processed_ops_details.append(op_detail)

        elif op == 'I':
            if alignment_started:
                # Internal Insertion: Mark its position *at* the current ref pos
                op_detail['viz_start'] = current_ref_pos
                op_detail['viz_width'] = length # Store length for label/drawing hint
                processed_ops_details.append(op_detail)
            else:
                # Leading Insertion: Treat like Soft clip for visualization position
                # Draw it to the left of start_pos
                current_viz_prefix_pos -= length # Decrement the prefix start position
                op_detail['viz_start'] = current_viz_prefix_pos
                op_detail['viz_width'] = length
                op_detail['is_leading_clip_or_ins'] = True # Mark for drawing pass
                processed_ops_details.append(op_detail)

        elif op in ['S', 'H']:
            if alignment_started:
                # Trailing Clip: Mark position relative to the last ref base
                op_detail['viz_start'] = last_ref_consuming_pos
                op_detail['viz_width'] = length
                op_detail['is_trailing_clip'] = True # Mark for drawing pass
                processed_ops_details.append(op_detail)
                 # Note: We'll need to adjust start based on previous trailing clips when drawing
            else:
                # Leading Clip: Draw it to the left of start_pos
                current_viz_prefix_pos -= length # Decrement the prefix start position
                op_detail['viz_start'] = current_viz_prefix_pos
                op_detail['viz_width'] = length
                op_detail['is_leading_clip_or_ins'] = True # Mark for drawing pass
                processed_ops_details.append(op_detail)
        # P (Padding) is usually ignored in reference visualization

    # --- Pass 2: Draw elements ---
    min_coord = float('inf')
    max_coord = float('-inf')
    trailing_clip_offset = 0 # Keep track of space used by trailing clips

    for op_detail in processed_ops_details:
        op = op_detail['op']
        length = op_detail['len']
        viz_start = op_detail['viz_start']
        viz_width = op_detail['viz_width']
        color = color_map.get(op, 'gray')
        edge_color = 'black'
        alpha = 0.75

        # Update coordinate bounds
        min_coord = min(min_coord, viz_start)


        if op_detail.get('is_leading_clip_or_ins'):
            rect = patches.Rectangle(
                (viz_start, y_pos - clip_height / 2), viz_width, clip_height,
                linewidth=0.5, edgecolor=edge_color, facecolor=color, alpha=alpha
            )
            ax.add_patch(rect)
            ax.text(viz_start + viz_width / 2, y_pos, op, ha='center', va='center', fontsize=7, color='black')
            max_coord = max(max_coord, viz_start + viz_width)

        elif op_detail.get('is_trailing_clip'):
            actual_start = last_ref_consuming_pos + trailing_clip_offset
            rect = patches.Rectangle(
                (actual_start, y_pos - clip_height / 2), viz_width, clip_height,
                linewidth=0.5, edgecolor=edge_color, facecolor=color, alpha=alpha
            )
            ax.add_patch(rect)
            ax.text(actual_start + viz_width / 2, y_pos, op, ha='center', va='center', fontsize=7, color='black')
            trailing_clip_offset += viz_width
            max_coord = max(max_coord, actual_start + viz_width)

        elif op in ['M', '=', 'X']:
            rect = patches.Rectangle(
                (viz_start, y_pos - height / 2), viz_width, height,
                linewidth=1, edgecolor=edge_color, facecolor=color, alpha=alpha
            )
            ax.add_patch(rect)
            ax.text(viz_start + viz_width / 2, y_pos, op, ha='center', va='center', fontsize=8, color='white' if op != 'S' else 'black')
            max_coord = max(max_coord, viz_start + viz_width)

        elif op == 'D':
            # Draw Deletion as a line or thinner rectangle
            rect = patches.Rectangle(
                (viz_start, y_pos - height / 2 * 0.6), viz_width, height * 0.6, # Thinner
                linewidth=0, edgecolor=edge_color, facecolor=color, alpha=alpha # No edge?
            )
            # Or draw as a simple line:
            # ax.plot([viz_start, viz_start + viz_width], [y_pos, y_pos], color=color, linewidth=2)
            ax.add_patch(rect)
            ax.text(viz_start + viz_width / 2, y_pos, 'D', ha='center', va='center', fontsize=7, color='white')
            max_coord = max(max_coord, viz_start + viz_width)

        elif op == 'N': # Skipped Region (like introns)
             # Draw as a dashed line connecting the ends
            ax.plot([viz_start, viz_start + viz_width], [y_pos, y_pos],
                    color=color, linestyle='--', linewidth=1)
            # Optionally add a thin box beneath the line
            rect = patches.Rectangle(
                 (viz_start, y_pos - height / 8), viz_width, height / 4,
                 linewidth=0, facecolor=color, alpha=0.3
            )
            ax.add_patch(rect)
            # ax.text(viz_start + viz_width / 2, y_pos + 0.01, 'N', ha='center', va='bottom', fontsize=7, color='black') # Text above line
            max_coord = max(max_coord, viz_start + viz_width)


        elif op == 'I': # Internal Insertion
            # Draw above the line at the insertion point
            # Use a smaller width visually as it doesn't consume reference
            ins_viz_width = max(1, length * 0.1) # Arbitrary visual width, not scale of ref
            ins_viz_width = min(ins_viz_width, 5) # Cap visual width
            rect = patches.Rectangle(
                (viz_start - ins_viz_width / 2, y_pos + insertion_offset - insertion_height / 2),
                ins_viz_width, insertion_height,
                linewidth=0.5, edgecolor=edge_color, facecolor=color, alpha=alpha
            )
            ax.add_patch(rect)
            ax.text(viz_start, y_pos + insertion_offset, f'I{length}',
                    ha='center', va='center', fontsize=7, color='black')
            # Insertion does not advance max_coord based on reference

    # Add CIGAR label (adjust position based on min_coord)
    label_pos_x = min(start_pos, min_coord) # Place label left of the entire visualization
    ax.text(label_pos_x - 10, y_pos, "CIGAR:", ha='right', va='center', fontsize=9, weight='bold')

def modify_cigar_for_svs(cigar_string,read_length, match_threshold=25):
    """
    Merges SV operations (D/I/N) separated by small matches (< threshold), and converts short SVs to dominant op.
    """
    cigar_ops = re.findall(r'(\d+)([MIDNSHP=X])', cigar_string)
    cigar_ops = [(int(length), op) for length, op in cigar_ops]

    new_cigar = []
    i = 0

    while i < len(cigar_ops):
        length, op = cigar_ops[i]

        # If it's a SV operation, begin grouping
        if op in "DIN":
            group = [(length, op)]
            j = i + 1

            while j < len(cigar_ops):
                next_len, next_op = cigar_ops[j]
                if next_op == 'M' and next_len < match_threshold:
                    group.append((next_len, next_op))
                    j += 1
                elif next_op in "DIN":
                    group.append((next_len, next_op))
                    j += 1
                else:
                    break

            # Decide dominant SV op (D/I/N) with highest total bp
            sv_lengths = Counter()
            for l, o in group:
                if o in "DIN":
                    sv_lengths[o] += l

            if sv_lengths:
                dominant_op = sv_lengths.most_common(1)[0][0]
                total_len = sum(l for l, o in group)
                new_cigar.append((total_len, dominant_op))
            else:
                # unlikely, but fallback
                new_cigar.extend(group)

            i = j
        else:
            new_cigar.append((length, op))
            i += 1

    new_cigar_string = ''.join(f"{l}{o}" for l, o in new_cigar)
    new_cigar_ops = re.findall(r'(\d+)([MIDNSHP=X])', new_cigar_string)
    
    # Detect if large match is before the first half of the read as portion
    first_match = 0
    last_match = 0
    large_match_before_half = 0
    large_match_after_half = 0
    reverse_cigar = False
    for ops in new_cigar_ops:
        while ops[1] == "M" and not first_match:
            first_match = 1
            large_match_before_half += int(ops[0])
            
    for ops in new_cigar_ops[::-1]:
        if ops[1] == "M" and not last_match:
            last_match = 1
            large_match_after_half += int(ops[0])
            
    if large_match_before_half < large_match_after_half:
        new_cigar_ops = new_cigar_ops[::-1]
        reverse_cigar = True
            
    
    # check for soft clips
    counter = 0
    while counter <= read_length:
        for i in range(len(new_cigar_ops)):
            length, op = new_cigar_ops[i]
            if op == "M" and (i + 1) < len(new_cigar_ops):
                next_length, next_op = new_cigar_ops[i+1]
                if counter + int(length) + int(next_length) <= read_length:
                    counter += int(length)
                    
                else:
                    counter += int(length)
                    new_cigar_ops[i+1] = [str(read_length - counter), "S"]
                    counter += int(next_length)
            
            if counter > read_length:
                break
                    
        if i == len(new_cigar_ops) - 1:
            break
                    
    # delete operations after the soft clip
    index = -1
    for i in range(len(new_cigar_ops)):
        if new_cigar_ops[i][1] == "S" and cigar_ops[i][1] != "S":
            index = i
            break
    if index != -1:
        new_cigar_ops = new_cigar_ops[:index+1]
        
    if reverse_cigar:
        new_cigar_ops = new_cigar_ops[::-1]
    
    soft_clip_cigar = ''.join(f"{l}{o}" for l, o in new_cigar_ops)

    return new_cigar_string, soft_clip_cigar

def detect_sv_from_cigar_coverage(cigar_string, start_pos, depth_dict,read_length, padding=50):
    """
    Analyze coverage around indels identified in CIGAR strings to detect SVs.
    
    Args:
        cigar_string: The CIGAR string to analyze
        start_pos: Starting position on the reference
        depth_dict: Dictionary mapping reference positions to their depth
        padding: Number of bases to check before and after the indel
        
    Returns:
        List of detected SVs
    """
    
    if not cigar_string:
        return []
    
    # Parse CIGAR string
    modified_cigar_string, _ = modify_cigar_for_svs(cigar_string, read_length)
    cigar_ops = re.findall(r'(\d+)([MIDNSHP=X])', modified_cigar_string)
    
    current_pos = start_pos
    min_sv_size = 10  # Minimum SV size to report
    detected_svs = []
    
    # Track reference positions of each CIGAR operation
    for length, op in cigar_ops:
        length = int(length)
        
        # Only interested in indels (I and D) and skipped regions (N)
        if op in ['I', 'D', 'N', 'S'] and length >= min_sv_size:
            sv_type = 'insertion' if op == 'I' else 'deletion' if op == 'D' else 'skipped_region' if op == 'N' else 'soft_clip' if op == 'S' else None
            
            # Determine SV region (different for insertions vs deletions/skipped)
            if op == 'I':
                # Insertions don't consume reference, so position is the base before
                sv_start = current_pos - 1
                sv_end = current_pos
            else:
                # Deletions and skipped regions consume reference
                sv_start = current_pos
                sv_end = current_pos + length - 1
            
            # Get coverage context around the indel
            # More sensitive: use wider context to better detect events
            context_start = max(sv_start - padding*2, 0)
            context_end = sv_end + padding*2
            
            # Extract depths in context region
            context_positions = sorted([pos for pos in depth_dict if context_start <= pos <= context_end])
            
            if not context_positions:
                continue
                
            context_depths = [depth_dict[pos] for pos in context_positions]
            
            # Calculate local coverage statistics
            local_avg_depth = np.mean(context_depths)
            local_std_depth = np.std(context_depths)
            
            # Dynamic threshold based on local coverage
            # For deletions: should be lower than average
            # For insertions: can have average or higher coverage
            if op in ['D', 'N', 'S']:
                # For deletions, check if depth inside deletion is significantly lower
                deletion_positions = [pos for pos in context_positions if sv_start <= pos <= sv_end]
                if not deletion_positions:
                    continue
                    
                deletion_depths = [depth_dict[pos] for pos in deletion_positions]
                deletion_avg_depth = np.mean(deletion_depths)
                
                # More sensitive: relax the threshold to detect more subtle deletions
                deletion_threshold = local_avg_depth * 0.7
                
                is_sv = deletion_avg_depth < deletion_threshold
                
                # Calculate depth ratio for confidence
                depth_ratio = deletion_avg_depth / local_avg_depth if local_avg_depth > 0 else 0
                
                if is_sv:
                    detected_svs.append({
                        "start": sv_start,
                        "end": sv_end,
                        "length": length,
                        "type": sv_type,
                        "local_avg_depth": local_avg_depth,
                        "sv_avg_depth": deletion_avg_depth,
                        "depth_ratio": depth_ratio,
                        "cigar_op": f"{length}{op}"
                    })
            
            elif op in ['I', 'S']:
                # For insertions, we check for coverage consistency around the insertion point
                left_context = [pos for pos in context_positions if context_start <= pos < sv_start]
                right_context = [pos for pos in context_positions if sv_end < pos <= context_end]
                
                if not left_context or not right_context:
                    continue
                    
                left_depths = [depth_dict[pos] for pos in left_context]
                right_depths = [depth_dict[pos] for pos in right_context]
                
                left_avg = np.mean(left_depths)
                right_avg = np.mean(right_depths)
                
                # Insertion should maintain similar coverage on both sides
                # They may have increased coverage if insertion is partially aligned
                avg_side_coverage = (left_avg + right_avg) / 2
                coverage_diff = abs(left_avg - right_avg)
                
                # More sensitive: allow higher difference
                is_sv = coverage_diff < local_avg_depth * 0.5  # Allow up to 50% difference
                
                if is_sv:
                    detected_svs.append({
                        "start": sv_start,
                        "end": sv_end,
                        "length": length,
                        "type": sv_type,
                        "local_avg_depth": local_avg_depth,
                        "left_avg_depth": left_avg,
                        "right_avg_depth": right_avg,
                        "coverage_diff": coverage_diff,
                        "cigar_op": f"{length}{op}"
                    })
        
        # Update reference position
        if op in ['M', 'D', 'N', '=', 'X']:
            current_pos += length
    
    return detected_svs

def calculate_read_coverage_range(read):
    """
    Calculate the raw coverage range of a read, accounting for all CIGAR operations.
    
    Args:
        read: pysam.AlignedSegment object
        
    Returns:
        tuple: (read_pos_start, read_pos_end) - the full coverage range including soft clips
    """
    
    if not read.cigarstring or not read.is_mapped:
        return read.reference_start, read.reference_start + read.query_length
    
    # Get basic reference coordinates
    ref_start = read.reference_start
    ref_end = read.reference_end if read.reference_end else ref_start
    
    # Get full read length from CIGAR
    read_length = sum([length for op, length in read.cigartuples if op in (0, 1, 4, 7, 8)])
    
    # Parse CIGAR operations to find soft clips
    cigar_tuples = read.cigartuples
    if not cigar_tuples:
        return ref_start, ref_start + read_length
    
    # Adjust start position for soft clips at beginning
    if cigar_tuples[0][0] == 4:  # Soft clip at start
        ref_start = ref_start - cigar_tuples[0][1]
    
    # Adjust end position for soft clips at end
    if cigar_tuples[-1][0] == 4:  # Soft clip at end
        ref_end = ref_end + cigar_tuples[-1][1]
    
    return ref_start, ref_end

def call_svs_using_depth_graphs(reads_near_alignments, to_be_sv_called_reads, reference, output_folder):
    """
    Call structural variants (SVs) using depth graphs created from reads near alignments.
    
    Args:
        reads_near_alignments: Dictionary where keys are indices from to_be_sv_called_reads and 
                              values are lists of reads from the BAM file that are in the range
                              of (alignment start - 300) to (alignment end + 300)
        to_be_sv_called_reads: DataFrame containing reads with alignments
        
    Returns:
        DataFrame with SV calling results
    """
    
    # Set high-quality figure defaults
    plt.rcParams['figure.dpi'] = 120
    plt.rcParams['figure.figsize'] = (14, 12)  # Increased height for additional subplot
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.style.use('seaborn-v0_8-whitegrid')
    
    sv_results = []
    os.makedirs(f"{output_folder}/sv_plots", exist_ok=True)
    os.makedirs(f"{output_folder}/sv_reports", exist_ok=True)
    
    reference_file = load_human_reference_genome(reference)
    for seq in reference_file:
        reference_seq = seq.seq
    
    # Process each read in to_be_sv_called_reads
    for idx, row in to_be_sv_called_reads.iterrows():
        read_id = row["enum"]
        
        if read_id not in reads_near_alignments:
            continue
            
        # Get the reads near this alignment
        nearby_reads = reads_near_alignments[read_id]
        
        # Get alignment position and CIGAR string of the main read
        start_pos = int(row["read/frag_gt_idx"])
        end_pos = start_pos + int(row["read/frag_gt_alignment_index"][0][-1][-1]) - int(row["read/frag_gt_alignment_index"][0][0][0])
        main_cigar = row["alignment_cigar_str"]
        
        # Get ground truth CIGAR string
        gt_cigar = row["read_gt_cigar_str"]
        
        # Create position range with padding
        pos_range = range(start_pos - 300, end_pos + 300)
        
        # Calculate read depth at each position
        depth_dict = defaultdict(int)
        read_pos_counter_dict = defaultdict(int)
        pos_base_dict = defaultdict(lambda: defaultdict(int))
        
        # Collect SVs from CIGAR strings and aligned pairs
        cigar_sv_regions = []
        aligned_pairs_sv_regions = []
        
        # Create detailed alignment report
        alignment_report = []
        
        for read in nearby_reads:
            # Get the alignment positions for this read
            if read.cigarstring and read.is_mapped:
                read_start = read.reference_start
                read_end = read.reference_end if read.reference_end else read_start + read.query_length
            
            else:
                read_start = None
                read_end = None
            
            read_pos_start, read_pos_end = calculate_read_coverage_range(read)
            print(f"Read CIGAR: {read.cigarstring}")
            print(f"Read Start and End: {read_start} {read_end}")
            print(f"Modified Read Start and End: {read_pos_start} {read_pos_end}")
            print("-"*100)
            if read_pos_start is not None and read_pos_end is not None:
                for pos in range(read_pos_start, read_pos_end):
                    if pos in pos_range:
                        read_pos_counter_dict[pos] += 1
            
            # Calculate depth contribution
            # Using exact aligned positions instead of a simple range
            ref_positions = read.get_reference_positions()
            read_pos_to_base_dict = {k:v for k,v in zip(read.positions, read.query_alignment_sequence)}
            for pos in ref_positions:
                if pos in pos_range and pos in read.positions:
                    if (read_start and read_end) and (read_start <= pos <= read_end):
                        depth_dict[pos] += 1
                        pos_base_dict[pos][read_pos_to_base_dict[pos]] += 1
                                        
            # create consensus sequence where empty positions are -
            consensus_seq = ""
            for pos in pos_range:
                if pos in pos_base_dict:
                    consensus_seq += max(pos_base_dict[pos], key=pos_base_dict[pos].get)
                else:
                    consensus_seq += "-"
                    
            for pos in pos_range:
                if pos not in pos_base_dict:
                    depth_dict[pos] = 0
                
                if pos not in read_pos_counter_dict:
                    read_pos_counter_dict[pos] = 0
                    
            if read.cigarstring:
                print("-"*100)
                print("CIGAR STRING:", read.cigarstring)
                print("-"*100)
                print(format_alignment(reference_seq[read.reference_start:read.reference_start+read.query_length], 
                                       read.query_sequence, 
                                       read.reference_start - ((read.reference_start // 1000) * 1000), 
                                       read.cigarstring))
                print("-"*100)
        
            # Extract SVs from CIGAR string
            svs_from_cigar = extract_sv_from_cigar(read)
            for sv in svs_from_cigar:
                if sv['start'] in pos_range or sv['end'] in pos_range:
                    cigar_sv_regions.append(sv)
            
            # Analyze aligned pairs for more complex SV detection
            alignment_analysis = analyze_aligned_pairs(read)
            
            # Add alignment details to report
            alignment_report.append({
                'read_id': read.query_name,
                'read_length': read.query_length,
                'start_pos': read_start,
                'end_pos': read_end,
                'mapping_quality': read.mapping_quality,
                'cigar_string': read.cigarstring,
                'is_proper_pair': read.is_proper_pair,
                'alignment_analysis': alignment_analysis
            })
            
            # Extract SVs from aligned pairs analysis
            if 'insertions' in alignment_analysis:
                for ins in alignment_analysis['insertions']:
                    if ins['start'] in pos_range:
                        aligned_pairs_sv_regions.append({
                            'type': ins['type'],
                            'start': ins['start'],
                            'end': ins['start'] + 1,  # Insertions don't extend on reference
                            'length': ins['length'],
                            'sequence': ins.get('query_bases', '')
                        })
            
            if 'deletions' in alignment_analysis:
                for deletion in alignment_analysis['deletions']:
                    if deletion['start'] in pos_range:
                        aligned_pairs_sv_regions.append({
                            'type': deletion['type'],
                            'start': deletion['start'],
                            'end': deletion['start'] + deletion['length'],
                            'length': deletion['length'],
                            'sequence': deletion.get('ref_bases', '')
                        })
            
            if 'ref_gaps' in alignment_analysis:
                for gap in alignment_analysis['ref_gaps']:
                    if gap['start'] in pos_range or gap['end'] in pos_range:
                        aligned_pairs_sv_regions.append({
                            'type': 'large_deletion',
                            'start': gap['start'],
                            'end': gap['end'],
                            'length': gap['length']
                        })
        
        ratio_depth_dict = {}
        for pos in depth_dict:
            if pos in read_pos_counter_dict and read_pos_counter_dict[pos] != 0:
                ratio_depth_dict[pos] = depth_dict[pos] / read_pos_counter_dict[pos]
            else:
                ratio_depth_dict[pos] = depth_dict[pos]  # or some other default value
        # Print comparison between reference and consensus sequence
        extended_start = max(0, start_pos - 300) 
        extended_end = min(len(reference_seq), end_pos + 300)

        print("\nReference vs Consensus Comparison (Extended):")

        for i in range(extended_start, extended_end, 100):
            end_i = min(i + 100, extended_end)
            pos_str = f"Position: {i}-{end_i-1}"
            print(f"{pos_str}")
            
            # Get reference chunk
            ref_chunk = reference_seq[i:end_i]
            
            # Create consensus chunk with placeholders for extended regions
            cons_chunk = ""
            for pos in range(i, end_i):
                if pos in pos_base_dict:
                    cons_chunk += max(pos_base_dict[pos], key=pos_base_dict[pos].get)
                else:
                    cons_chunk += "-"
            
            # Print sequences
            print(f"Reference: {ref_chunk}")
            print(f"Consensus: {cons_chunk}")
            
            # Print match/mismatch indicators
            comparison = ""
            for j in range(len(ref_chunk)):
                ref_char = ref_chunk[j]
                cons_char = cons_chunk[j]
                if cons_char == "?":
                    comparison += " "  # No comparison possible
                else:
                    comparison += "|" if ref_char == cons_char else "."
            
            print(f"Match:     {comparison}\n")

        
        # Save alignment report
        alignment_report_df = pd.DataFrame(alignment_report)
        report_path = f"{output_folder}/sv_reports/alignment_report_{read_id}.csv"
        if not alignment_report_df.empty:
            alignment_report_df.to_csv(report_path, index=False)
        
        # Create depth array
        positions = sorted(depth_dict.keys())
        depths = [depth_dict[pos] for pos in positions]
        
        # Create ratio depth array
        ratio_positions = sorted(ratio_depth_dict.keys())
        ratio_depths = [ratio_depth_dict[pos] for pos in ratio_positions]
        
        # Skip if insufficient data
        if not positions:
            continue
        
        # Calculate average depth and standard deviation
        avg_depth = np.mean(depths)
        std_depth = np.std(depths)
        
        print(f"Read {read_id}: avg_depth={avg_depth:.2f}, std_depth={std_depth:.2f}")
        
        # =====================================================================
        # ANALYZE CIGAR STRING FOR SVs WITH DYNAMIC COVERAGE THRESHOLDS
        # =====================================================================
        cigar_coverage_sv_regions = detect_sv_from_cigar_coverage(main_cigar, start_pos, depth_dict,len(row["read_gt"]), padding=50)
        print(f"Read {read_id}: Found {len(cigar_coverage_sv_regions)} SV regions based on CIGAR analysis")
        
        # Merge all SV sources into one list
        sv_regions = cigar_coverage_sv_regions.copy()
        
        # # Add SVs detected from other methods
        # for sv in cigar_sv_regions:
        #     sv_regions.append({
        #         "start": sv["start"],
        #         "end": sv["end"],
        #         "min_depth": depth_dict[sv["start"]] if sv["start"] in depth_dict else 0,
        #         "length": sv["length"],
        #         "type": sv["type"],
        #         "detection_method": "cigar_extract"
        #     })
            
        # for sv in aligned_pairs_sv_regions:
        #     sv_regions.append({
        #         "start": sv["start"],
        #         "end": sv["end"],
        #         "min_depth": depth_dict[sv["start"]] if sv["start"] in depth_dict else 0,
        #         "length": sv["length"],
        #         "type": sv["type"],
        #         "sequence": sv.get("sequence", ""),
        #         "detection_method": "aligned_pairs"
        #     })
        
        # Generate enhanced depth plot with CIGAR visualization
        fig = plt.figure(figsize=(14, 14))  # Increased height to accommodate the additional panel
        gs = gridspec.GridSpec(5, 1, height_ratios=[3, 1, 1, 1, 1])  # Added one more row for ratio plot
        
        # Create depth plot in the top panel
        ax1 = plt.subplot(gs[0])
        ax1.plot(positions, depths, '-', color='blue', linewidth=1.5, alpha=0.8, label='Read Depth')
        ax1.axhline(y=avg_depth, color='green', linestyle='--', linewidth=2, label=f'Avg Depth: {avg_depth:.2f}')
        
        # Highlight SV regions with different colors based on type
        color_map = {
            "deletion": "red",
            "insertion": "purple",
            "soft_clip_start": "orange",
            "soft_clip_end": "yellow",
            "large_deletion": "darkred",
            "skipped_region": "brown"
        }
        
        # Sort SVs by start position for better label placement
        sorted_sv_regions = sorted(sv_regions, key=lambda x: x["start"])
        
        # Group overlapping SVs that likely represent the same variant
        merged_sv_regions = []
        current_group = None
        
        # Simple check to merge identical SVs
        def is_same_sv(sv1, sv2):
            """Check if two SVs are exactly the same"""
            return (sv1["type"] == sv2["type"] and
                    sv1["start"] == sv2["start"] and
                    sv1["end"] == sv2["end"])
            
        for sv in sorted_sv_regions:
            if current_group is None:
                # Start a new group
                current_group = sv.copy()
                current_group["detection_methods"] = [sv.get("detection_method", "cigar_coverage")]
            elif is_same_sv(current_group, sv):
                # Add to current group
                detection_method = sv.get("detection_method", "cigar_coverage")
                if detection_method not in current_group["detection_methods"]:
                    current_group["detection_methods"].append(detection_method)
                
                # Keep the more precise coordinates if available
                if "sv_avg_depth" in sv and "depth_ratio" in sv:
                    current_group["sv_avg_depth"] = sv["sv_avg_depth"]
                    current_group["depth_ratio"] = sv["depth_ratio"]
                    
                if "coverage_diff" in sv:
                    current_group["coverage_diff"] = sv["coverage_diff"]
                    
                if "cigar_op" in sv and "cigar_op" not in current_group:
                    current_group["cigar_op"] = sv["cigar_op"]
                    
                if "sequence" in sv and "sequence" not in current_group:
                    current_group["sequence"] = sv["sequence"]
            else:
                # Finish current group and start a new one
                merged_sv_regions.append(current_group)
                current_group = sv.copy()
                current_group["detection_methods"] = [sv.get("detection_method", "cigar_coverage")]
                
        # Add the last group if exists
        if current_group is not None:
            merged_sv_regions.append(current_group)
            
        # Create a list to store SV information for the legend table
        sv_legend_data = []
        
        # Use numbers instead of text annotations directly on the plot
        for i, sv in enumerate(merged_sv_regions):
            sv_id = i + 1  # 1-based numbering for readability
            color = color_map.get(sv["type"], "red")
            
            # Fill SV region with color
            ax1.axvspan(sv["start"], sv["end"], alpha=0.3, color=color)
            
            # Calculate center position and set label
            x_center = (sv["start"] + sv["end"]) / 2
            
            y_position = max(depths) * 0.9  # Position label below top of plot
            
            # Format detection methods as comma-separated list
            methods_str = ", ".join(sv["detection_methods"])
            
            # Create label text
            label_text = f"{sv['type'].upper()} ({sv['length']}bp)"
            if "cigar_op" in sv:
                label_text += f"\nCIGAR: {sv['cigar_op']}"
            if "depth_ratio" in sv:
                label_text += f"\nDepth: {sv['depth_ratio']:.2f}"
                
            # Add text label directly on plot
            ax1.text(
                x_center, y_position,
                label_text,
                ha='center',
                va='center',
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor=color, boxstyle='round'),
                zorder=5
            )
            
            # Add depth ratio or coverage diff if available
            if "depth_ratio" in sv:
                # Already added to label
                pass
            elif "coverage_diff" in sv:
                # Already added to label
                pass
            
            # Add CIGAR operation if available
            # Already added to label
            
            # If dynamic threshold was used, add a horizontal line showing it
            if "sv_avg_depth" in sv and sv["type"] == "deletion":
                deletion_threshold = sv["local_avg_depth"] * 0.5
                ax1.axhline(
                    y=deletion_threshold, 
                    color='red', 
                    linestyle=':', 
                    linewidth=1.5, 
                    alpha=0.7,
                    xmin=(sv["start"] - min(positions)) / (max(positions) - min(positions)),
                    xmax=(sv["end"] - min(positions)) / (max(positions) - min(positions)),
                    label=f'Deletion Threshold: {deletion_threshold:.2f}'
                )
        
        # Add vertical lines for start and end of main read alignment
        ax1.axvline(x=start_pos, color='green', linestyle='-', linewidth=2, alpha=0.7, label='Alignment Start')
        ax1.axvline(x=end_pos, color='red', linestyle='-', linewidth=2, alpha=0.7, label='Alignment End')
        
        # Configure axis 1
        ax1.set_xlabel('Reference Position', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Read Depth', fontsize=12, fontweight='bold')
        ax1.set_title(f'Depth Graph for Read {read_id}\nAlignment: {start_pos}-{end_pos}', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', frameon=True, framealpha=0.7)
        ax1.set_xlim(min(positions), max(positions))
        
        # Create ratio depth plot as second panel
        ax_ratio = plt.subplot(gs[1], sharex=ax1)
        ax_ratio.plot(ratio_positions, ratio_depths, '-', color='purple', linewidth=1.5, 
                     alpha=0.8, label='Aligned/Coverage Ratio')
        ax_ratio.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, 
                        label='Equal Ratio (1.0)')
        ax_ratio.set_ylim(0, max(2, max(ratio_depths) * 1.1))  # Adjust upper limit dynamically
        ax_ratio.set_ylabel('Depth Ratio', fontsize=10, fontweight='bold')
        ax_ratio.set_title('Ratio of Aligned Depth to Read Coverage', fontsize=9)
        ax_ratio.grid(True, alpha=0.3)
        ax_ratio.legend(loc='upper right', frameon=True, framealpha=0.7, fontsize=8)
        
        # Create CIGAR visualization in the middle panel (aligned CIGAR)
        ax2 = plt.subplot(gs[2], sharex=ax1)  # Changed from gs[1] to gs[2]
        cigar_ops = re.findall(r'(\d+)([MIDNSHP=X])', main_cigar)
        if cigar_ops[0][1] == "S":
            all_ops_lengths = [int(op[0]) for op in cigar_ops[1:]]
            sum_of_ops_lengths = sum(all_ops_lengths)
            visualize_cigar(main_cigar, end_pos - sum_of_ops_lengths, ax2)
        else:
            visualize_cigar(main_cigar, start_pos, ax2)
        ax2.set_yticks([])
        ax2.set_ylabel('Aligned CIGAR', fontsize=12, fontweight='bold')
        ax2.set_title(f'Aligned CIGAR: {main_cigar}', fontsize=10)
        
        # Create modified CIGAR visualization
        ax3 = plt.subplot(gs[3], sharex=ax1)  # Changed from gs[2] to gs[3]
        _ ,modified_cigar = modify_cigar_for_svs(main_cigar, len(row["read_gt"]))
        cigar_ops = re.findall(r'(\d+)([MIDNSHP=X])', modified_cigar)
        if cigar_ops[0][1] == "S":
            all_ops_lengths = [int(op[0]) for op in cigar_ops[1:]]
            sum_of_ops_lengths = sum(all_ops_lengths)
            visualize_cigar(modified_cigar, end_pos - sum_of_ops_lengths, ax3)
        else:
            visualize_cigar(modified_cigar, start_pos, ax3)
        ax3.set_yticks([])
        ax3.set_ylabel('Modified CIGAR', fontsize=12, fontweight='bold')
        ax3.set_title(f'Modified CIGAR: {modified_cigar}', fontsize=10)
        
        # Create ground truth CIGAR visualization in the bottom panel
        ax4 = plt.subplot(gs[4], sharex=ax1)  # Changed from gs[3] to gs[4]
        cigar_ops = re.findall(r'(\d+)([MIDNSHP=X])', gt_cigar)
        if cigar_ops[0][1] == "S":
            all_ops_lengths = [int(op[0]) for op in cigar_ops[1:]]
            sum_of_ops_lengths = sum(all_ops_lengths)
            visualize_cigar(gt_cigar, end_pos - sum_of_ops_lengths, ax4)
        else:
            visualize_cigar(gt_cigar, start_pos, ax4)
        ax4.set_yticks([])
        ax4.set_ylabel('GT CIGAR', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Reference Position', fontsize=12, fontweight='bold')
        ax4.set_title(f'Ground Truth CIGAR: {gt_cigar}', fontsize=10)
        
        # Add color legend for CIGAR operations
        cigar_colors = {
            'M': 'blue',      # Match/mismatch
            'I': 'green',     # Insertion
            'D': 'red',       # Deletion
            'S': 'yellow',    # Soft clip
            'H': 'purple',    # Hard clip
            'N': 'brown',     # Skipped region
        }
        
        # Create custom legend handles
        import matplotlib.patches as mpatches
        all_operations = set()
        for cigar in [main_cigar, modified_cigar, gt_cigar]:
            for op in cigar_colors:
                if op in cigar:
                    all_operations.add(op)
                    
        legend_handles = [mpatches.Patch(color=cigar_colors[op], label=op) 
                         for op in sorted(all_operations)]
        
        if legend_handles:
            ax4.legend(handles=legend_handles, loc='lower right', 
                     title='CIGAR Operations', ncol=len(legend_handles))
        
        plt.tight_layout()
        
        # Save plot to file with high dpi
        plot_path = f"{output_folder}/sv_plots/depth_plot_{read_id}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Create result entry for each SV
        for sv in merged_sv_regions:
            min_depth = sv.get("min_depth", sv.get("sv_avg_depth", 0))
            depth_ratio = sv.get("depth_ratio", 0)
            if min_depth > 0 and depth_ratio == 0 and avg_depth > 0:
                depth_ratio = min_depth / avg_depth
                
            sv_result = {
                "read_id": read_id,
                "read_sequence": row["read_gt"],
                "reference_position": f"{start_pos}-{end_pos}",
                "cigar_string": main_cigar,
                "sv_start": sv["start"],
                "sv_end": sv["end"],
                "sv_length": sv["length"],
                "sv_type": sv["type"],
                "avg_depth": avg_depth,
                "local_avg_depth": sv.get("local_avg_depth", 0),
                "depth_ratio": depth_ratio,
                "cigar_operation": sv.get("cigar_op", ""),
                "detection_method": ", ".join(sv.get("detection_methods", ["cigar_coverage"])),
                "plot_path": plot_path,
                "report_path": report_path,
                "sequence": sv.get("sequence", "")
            }
            sv_results.append(sv_result)
    
    # Create DataFrame from results
    sv_results_df = pd.DataFrame(sv_results)
    return sv_results_df