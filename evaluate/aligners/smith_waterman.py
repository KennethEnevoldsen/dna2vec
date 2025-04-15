import time
from Bio import Align
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


def calculate_smith_waterman_distance(
    string1,
    string2,
    match_score=2,  # do not change
    mismatch_penalty=-1,
    open_gap_penalty=-0.5, #TODO: change it to -1.5
    continue_gap_penalty=-0.1, #TODO: change it to -0.5
    debug=False,
):
    start = time.time()
    # Perform the Smith-Waterman alignment
    aligner = Align.PairwiseAligner()
    aligner.open_gap_score = open_gap_penalty
    aligner.extend_gap_score = continue_gap_penalty
    aligner.mismatch_score = mismatch_penalty
    aligner.match_score = match_score
    aligner.mode = "local"
    alignments = aligner.align(string1, string2)
    # alignments = pairwise2.align.localms(
    #     string1, string2, match_score, mismatch_penalty,
    #     open_gap_penalty, continue_gap_penalty
    # )
    if debug:
        print("Best match:")
        print(alignments[0])

    if alignments == [] or alignments is None:  # exponentially large
        return {
            "elapsed time": time.time() - start,
            "distance": float("inf"),
            "begins": [],
        }

    try:
        if len(alignments) < 1:
            return {
                "elapsed time": time.time() - start,
                "distance": float("inf"),
                "begins": [],
            }
    except OverflowError:  # too many matches
        pass

    alignment = alignments[0]
    alignment_score = None

    alignment_score = alignment.score
    # Calculate the Smith-Waterman distance
    smith_waterman_distance = -alignment_score
    begins = []
    begins.append(
        alignment.aligned[0][0][0]
    )  # target, first match, first index TODO what if many matches - rare at this length

    return {
        "elapsed time": time.time() - start,
        "distance": smith_waterman_distance,
        "begins": list(set(begins)),
        "alignment_str": alignment.__str__(),
        "alignment_indices": alignment.aligned
    }


def bwamem_align(
    all_candidate_strings: list[str],
    trained_positions: list[str],
    metadata_set: list[str],
    substring: list[str],
):
    """
    Currently implemented sequentially
    """

    total_time = 0
    refined_results = defaultdict(list)

    for long_string, train_pos, metadata in zip(
        all_candidate_strings, trained_positions, metadata_set
    ):

        returned_object = calculate_smith_waterman_distance(long_string,substring)
        total_time += returned_object["elapsed time"]

        for starting_sub_index in returned_object["begins"]:
            refined_results[returned_object["distance"]].append(
                (starting_sub_index, train_pos, metadata)
            )

    try:
        smallest_key = min(refined_results.keys())

    except ValueError:
        return [], [], [], total_time

    smallest_values = refined_results[smallest_key]

    identified_sub_indices = []
    identified_indices = []
    metadata_indices = []

    for term in smallest_values:
        identified_sub_indices.append(term[0])
        identified_indices.append(term[1])
        metadata_indices.append(term[2])

    return (
        identified_sub_indices,
        identified_indices,
        metadata_indices,
        total_time,
        smallest_key,
    )


def process_single_string(args: tuple):
    retrieved_fragment, read, train_pos, metadata = args
    returned_object = calculate_smith_waterman_distance(retrieved_fragment, 
                                                        read,
                                                        match_score=2,  # Keep as specified
                                                        mismatch_penalty=-1,  # Standard for nucleotide mismatches
                                                        open_gap_penalty=-2.5,  # Higher penalty for reference sequence (string1)
                                                        continue_gap_penalty=-0.02,  # Very low extension penalty for insertions
                                                        debug=False)
    # returned_object = calculate_smith_waterman_distance(retrieved_fragment, read)
    return (
        returned_object["distance"],
        returned_object["begins"],
        train_pos,
        metadata,
        returned_object["elapsed time"],
        retrieved_fragment,
        read,
        returned_object["alignment_str"],
        returned_object["alignment_indices"]
    )


def bwamem_align_parallel(
    all_candidate_strings: list[str],
    trained_positions: list[str],
    metadata_set: list[str],
    substring: str,
    orginal_read: str,
    max_workers: int = 50,
    file_path: str = "/home/shreyas/NLP/dna2vec/Results/streamed_results.jsonl",
):

    total_time = time.time()
    refined_results = defaultdict(list)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            executor.map(
                process_single_string,
                zip(
                    all_candidate_strings,
                    [substring] * len(all_candidate_strings),
                    trained_positions,
                    metadata_set,
                ),
            )
        )

    for distance, begins, train_pos, metadata, _, fragment, read, alignment_str,alignment_indices in results:
        for starting_sub_index in begins:
            refined_results[distance].append(
                (starting_sub_index, train_pos, metadata, fragment, read, alignment_str,alignment_indices)
            )

    try:
        smallest_key = min(refined_results.keys())
        print("SMALLEST KEY IS ", smallest_key)

        if smallest_key <= -495:

            print("SMALLEST KEY IS ", smallest_key)
            # write into a jsonlines file
            # print("WRITING INTO FILE")
            # with jsonlines.open(file_path, mode="a") as writer:
            #     for (
            #         starting_sub_index,
            #         train_pos,
            #         metadata,
            #         fragment,
            #         read,
            #     ) in refined_results[smallest_key]:
            #         writer.write(
            #             {
            #                 "distance": str(smallest_key),
            #                 "starting_sub_index": str(starting_sub_index),
            #                 "train_pos": train_pos,
            #                 "metadata": metadata,
            #                 "fragment": fragment,
            #                 "read": read,
            #             }
            #         )

    except ValueError:
        print("NO RESULTS FOUND")
        print(substring)
        print(trained_positions)
        return [], [], [], time.time() - total_time, None, None

    # dist_indices = defaultdict(list)
    # dist_meta = defaultdict(list)]
    distances = set()
    indices = set()
    index_to_trained_positions = dict()
    index_to_distance = dict()
    index_to_alignment_str = dict()
    index_to_alignment_indices = dict()
    for distance in refined_results:
        distances.add(distance)
        for term in refined_results[distance]:
            index = int(term[0]) + int(term[1])
            if index in indices:
                if index_to_distance[index][0] < distance:
                    continue
            else:
                indices.add(index)
            
            index_to_distance[index] = (distance, term[-4])
            index_to_trained_positions[index] = int(term[1])
            index_to_alignment_str[index] = term[-2]
            index_to_alignment_indices[index] = term[-1]
            # dist_meta[distances].append(meta)

    # Compute SW distance between subsequence and original read
    if orginal_read is None:
        orig_distance = -500
    else:

        orig_distance = calculate_smith_waterman_distance(orginal_read, substring)[
            "distance"
        ]

    # smallest_values = refined_results[smallest_key]

    # identified_sub_indices = []
    # identified_indices = []
    # metadata_indices = []

    # for term in smallest_values:
    #     identified_sub_indices.append(term[0])
    #     identified_indices.append(term[1])
    #     metadata_indices.append(term[2])

    # return (
    #     identified_sub_indices,
    #     identified_indices,
    #     metadata_indices,
    #     time.time() - total_time,
    #     smallest_key,
    # )

    return (
        distances,
        indices,
        refined_results,
        index_to_distance,
        orig_distance,
        index_to_trained_positions,
        index_to_alignment_str,
        index_to_alignment_indices,
        time.time() - total_time,
    )


def process_batch(batch):
    return [process_single_string(args) for args in batch]


def bwamem_align_parallel_process(
    all_candidate_strings: list[str],
    trained_positions: list[str],
    metadata_set: list[str],
    substring: str,
    batch_size: int = 5,
    max_workers: int = 10,
):
    total_time = time.time()
    refined_results = defaultdict(list)

    batches = [
        batch
        for batch in zip(
            all_candidate_strings,
            [substring] * len(all_candidate_strings),
            trained_positions,
            metadata_set,
        )
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        batch_results = list(
            executor.map(
                process_batch,
                [
                    batches[i : i + batch_size]
                    for i in range(0, len(batches), batch_size)
                ],
            )
        )

    for batch_result in batch_results:
        for results in batch_result:
            distance, begins, train_pos, metadata, _ = results
            for starting_sub_index in begins:
                refined_results[distance].append(
                    (starting_sub_index, train_pos, metadata)
                )

    if not refined_results:
        return [], [], [], time.time() - total_time

    smallest_key = min(refined_results.keys())
    smallest_values = refined_results[smallest_key]

    identified_sub_indices = []
    identified_indices = []
    metadata_indices = []

    for term in smallest_values:
        identified_sub_indices.append(term[0])
        identified_indices.append(term[1])
        metadata_indices.append(term[2])

    return (
        identified_sub_indices,
        identified_indices,
        metadata_indices,
        time.time() - total_time,
        smallest_key,
    )
