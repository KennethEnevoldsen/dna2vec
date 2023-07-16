import time
from Bio import pairwise2
from Bio.pairwise2 import format_alignment

def calculate_smith_waterman_distance(  string1,
                                        string2,
                                        match_score = 2,
                                        mismatch_penalty = -1,
                                        open_gap_penalty = -0.5,
                                        continue_gap_penalty = -0.1,
                                        debug = False):
    start = time.time()
    # Perform the Smith-Waterman alignment

    alignments = pairwise2.align.localms(
        string1, string2, match_score, mismatch_penalty, 
        open_gap_penalty, continue_gap_penalty
    )
    
    if debug:
        print("Best match:")
        print(format_alignment(*alignments[0]))
        
    # Extract the alignment score
    alignment_score = alignments[0].score
    # Calculate the Smith-Waterman distance
    smith_waterman_distance = -alignment_score
    
    begins = []
    for align in alignments:
        begins.append(align.start)
    
    return {"elapsed time": time.time() - start, 
            "distance": smith_waterman_distance, 
            "begins":begins}

###

def calculate_needleman_wunsch_distance(string1, 
                                        string2, 
                                        match_score = 2,
                                        mismatch_penalty = -1,
                                        open_gap_penalty = -0.5,
                                        continue_gap_penalty = -0.1,
                                        debug = False):
    start = time.time()

    # Perform the Needleman-Wunsch alignment
    alignments = pairwise2.align.globalms(
        string1, string2, match_score, mismatch_penalty, 
        open_gap_penalty, continue_gap_penalty
    )

    if debug:
        print("Best match:")
        print(format_alignment(*alignments[0]))
    # Extract the alignment score
    alignment_score = alignments[0].score
    begins = []
    for align in alignments:
        begins.append(align.start)
    # Calculate the Needleman-Wunsch distance
    needleman_wunsch_distance = -alignment_score
    
    return {"elapsed time": time.time() - start, 
            "distance": needleman_wunsch_distance, 
            "begins":begins}

