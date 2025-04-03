"""
Functions for simulating reads using ART as well as for reading in the simulated reads.
"""
import logging
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union
import pandas as pd
import pysam
from pysam.libcalignedsegment import AlignedSegment
from tqdm import tqdm
from dna2vec.utils import (
    download_human_reference_genome,
    get_cache_dir,
    load_human_reference_genome,
)

SEQUENCE_SYSTEMS = Literal[
    "GA1", "GA2", "HS10", "HS20", "HS25", "HSXn", "HSXt", "MinS", "MSv1", "MSv3", "NS50"
]


@dataclass
class ReadAndReference:
    """
    Class for storing a read and its reference.

    Attributes:
        read: The read.
        reference: The reference sequence as a string.
    """

    read: AlignedSegment
    reference: Union[str, None] = None
    id: Union[str, None] = None

    def is_mapped(self) -> bool:
        return self.reference is not None

    def get_pair_as_string(self) -> Tuple[str, Union[str, None]]:
        """
        Get the pair as a tuple of strings.
        """
        return (self.read.query_sequence, self.reference)


def _format_scientific_notation(number: float) -> str:
    return "0" + format(9e-05, ".10f").strip("0")


def simulate_reads_to_disk(
    n_reads_pr_amplicon: int,
    read_length: int,
    output_path: Path,
    reference_genome: Union[Path, None] = None,
    insertion_rate: float = 0.00009,
    deletion_rate: float = 0.00011,
    quality: Tuple[int, int] = (60, 93),
    sequencing_system: SEQUENCE_SYSTEMS = "HS20",
) -> Path:
    """
    Simulates unpaired reads using the ART command line interface.

    Args:
        n_reads_pr_amplicon: Number of reads to simulate per amplicon. An amplicon is a region of the reference genome.
        read_length: Length of the reads to simulate.
        reference_genome: Path to the reference genome to simulate reads from. If None it will download the human reference genome to the cache
            directory.
        output: Path to the output file. If None it will be saved to the cache directory.
        insertation_rate: Insertation rate for the simulation.
        deletion_rate: Deletion rate for the simulation.
        sequencing_system: Sequencing system to use for the simulation.
    """
    if reference_genome is None:
        reference_genome = download_human_reference_genome()

    logging.info(f"Simulating reads using ART")
    cmd = [
        "art_illumina",
        "-ss",
        sequencing_system,
        "-sam",
        "-i",
        str(reference_genome),
        "-l",
        str(read_length),
        "-c",
        str(n_reads_pr_amplicon),
        "-o",
        str(output_path),
        "--insRate",
        _format_scientific_notation(insertion_rate),
        "--delRate",
        _format_scientific_notation(deletion_rate),
        "--minQ",
        str(quality[0]),
        "--maxQ",
        str(quality[1]),
    ]
    logging.info(f"Running ART with the following command: {' '.join(cmd)}")

    subprocess.run(cmd, check=True)

    return output_path


def load_simulated_reads_from_disk(
    simulated_path: Path,
) -> list[AlignedSegment]:
    """
    Read in the simulated reads

    Args:
        simulated_path: Path to the simulated reads.

    Returns:
        List of aligned segments.
    """
    logging.info(f"Reading simulated reads from {simulated_path}")
    print(simulated_path)

    sam = simulated_path.with_suffix(".sam")

    # load using pysam
    aligned_segments = pysam.AlignmentFile(str(sam))
    return list(aligned_segments)

def load_real_reads_from_disk(
    bam_file: Path,
) -> list[AlignedSegment]:
    """
    Read in the simulated reads

    Args:
        simulated_path: Path to the simulated reads.

    Returns:
        List of aligned segments.
    """
    logging.info(f"Reading real reads from {bam_file}")
    print(bam_file)

    # load using pysam
    aligned_segments = pysam.AlignmentFile(str(bam_file), "rb")
    return aligned_segments


def map_reads_to_reference(
    reads: List[AlignedSegment],
    reference: Optional[Path] = None,
) -> List[ReadAndReference]:
    """
    Map a read to a reference genome.
    """
    reference = load_human_reference_genome(reference)

    id2read = defaultdict(list)
    unmapped_reads = []
    for read in reads:
        _id = read.query_name.split("-")[0]

        unmapped_read = ReadAndReference(read=read)
        id2read[_id].append(unmapped_read)
        unmapped_reads.append(unmapped_read)

    seq_offset = 0
    for seq in reference:
        matches = id2read[seq.id]
        for match in matches:
            read = match.read
            start = read.reference_start
            length = read.query_length
            original_sequence = seq.seq[start : start + length]
            match.reference = str(original_sequence)
            match.id = seq.id
            match.seq_offset = seq_offset
            assert read.query_sequence == read.seq
        seq_offset += len(seq.seq)
    return unmapped_reads

def map_real_reads_to_reference(
    reads: List[AlignedSegment],
    reference: Optional[Path] = None,
) -> List[ReadAndReference]:
    """
    Map a read to a reference genome.
    """
    reference = load_human_reference_genome(reference)

    id2read = defaultdict(list)
    unmapped_reads = []
    # for read in reads.fetch("chr2",1018500,2020000):
    #     _id = "chr2"

    #     if len(unmapped_reads) > 2000:
    #         break
    #     unmapped_read = ReadAndReference(read=read)
    #     id2read[_id].append(unmapped_read)
    #     unmapped_reads.append(unmapped_read)

    # === Step 1: Load chr2 deletions from VCF ===
    deletions = []
    vcf_path = "/home/yigit/codebase/dna2vec/human_deletions_GS_correct.vcf"
    with open(vcf_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            chrom = parts[0]
            if chrom != "2" and chrom != "chr2":  # Accept both notations
                continue
            start = int(parts[1])
            info = dict(x.split("=") for x in parts[7].split(";") if "=" in x)
            end = int(info["END"])
            svlen = int(info["SVLEN"])
            if svlen > 50:
                deletions.append((chrom, start, end))

    print(f"Loaded {len(deletions)} deletions on chr2.")
    
    # === Step 2: Extract reads with deletions, soft clips, or skipped regions ===
    for chrom, start, end in tqdm(deletions, desc="Processing chr2 deletions"):
        try:
            for read in tqdm(reads.fetch(chrom, start-249, end+249), desc="Processing reads"):
                if read.cigartuples:
                    has_deletion = any(op == 2 and length >= 1 for op, length in read.cigartuples)  # D
                    has_soft_clip = any(op == 4 and length >= 50 for op, length in read.cigartuples)  # S
                    has_skipped_region = any(op == 3 and length >= 1 for op, length in read.cigartuples)  # N
                    has_insertion = any(op == 1 and length >= 50 for op, length in read.cigartuples)  # I
                    
                    #if has_deletion or has_soft_clip or has_skipped_region or has_insertion:
                    if has_insertion:
                        _id = chrom
                        unmapped_read = ReadAndReference(read=read)
                        id2read[_id].append(unmapped_read)
                        unmapped_reads.append(unmapped_read)

                        if len(unmapped_reads) >= 500:
                            break
            if len(unmapped_reads) >= 500:
                break
        except ValueError:
            print(f"Region doesn't exist in BAM: {chrom}:{start}-{end}")
            continue
        
    seq_offset = 0
    for seq in reference:
        matches = id2read[seq.id]
        for match in matches:
            read = match.read
            start = read.reference_start
            length = read.query_length
            original_sequence = seq.seq[start : start + length]
            match.reference = str(original_sequence)
            match.id = seq.id
            match.seq_offset = seq_offset
            assert read.query_sequence == read.seq
        seq_offset += len(seq.seq)
    return unmapped_reads

def find_reference_reads(
    indices: List[int],
    reads: List[str],
    reference: Optional[Path] = None,
) -> List[ReadAndReference]:
    """
    Map a read to a reference genome.
    """
    reference = load_human_reference_genome(reference)
    index_query_df = pd.DataFrame({"index": indices, "query": reads})
    original_sequences = []
    for seq in reference:
        for idx, (loc, query) in index_query_df.iterrows():
            if pd.isna(loc):
                original_sequences.append(None)
            else:
                start = int(loc)
                length = len(query)
                original_sequence = seq.seq[start : start + length]
                original_sequences.append(original_sequence)
    return original_sequences


def _create_cache_path(
    n_reads_pr_amplicon: int,
    read_length: int,
    insertion_rate: float,
    deletion_rate: float,
    quality: Tuple[int, int],
    sequencing_system: SEQUENCE_SYSTEMS,
) -> Path:
    file_name_stubs = [
        "reads",
        f"{n_reads_pr_amplicon}",
        f"{read_length}",
        f"IR{str(insertion_rate).replace('.', '-dot-')}",
        f"DR{str(deletion_rate).replace('.', '-dot-')}",
        f"Q{str(quality[0])}-{str(quality[1])}",
        f"{sequencing_system}",
    ]

    file_name = "_".join(file_name_stubs)
    return get_cache_dir() / "simulated_reads" / file_name


def simulate_mapped_reads(
    n_reads_pr_amplicon: int,
    read_length: int,
    insertion_rate: float = 0.00009,
    deletion_rate: float = 0.00011,
    sequencing_system: SEQUENCE_SYSTEMS = "HS20",
    reference_genome: Union[Path, None] = None,
    quality: Tuple[int, int] = (60, 93),
):
    """
    Simulates reads and maps them to the reference genome.
    """

    output_path = _create_cache_path(
        n_reads_pr_amplicon=n_reads_pr_amplicon,
        read_length=read_length,
        insertion_rate=insertion_rate,
        deletion_rate=deletion_rate,
        sequencing_system=sequencing_system,
        quality=quality,
    )

    output_path.mkdir(parents=True, exist_ok=True)

    # if not output_path.with_suffix(".sam").exists():
    simulated_reads = simulate_reads_to_disk(
        n_reads_pr_amplicon=n_reads_pr_amplicon,
        read_length=read_length,
        output_path=output_path,
        reference_genome=reference_genome,
        insertion_rate=insertion_rate,
        deletion_rate=deletion_rate,
        sequencing_system=sequencing_system,
        quality=quality,
    )
    # else:
    #     logging.info(
    #         f"Simulated reads already exists at {output_path}. Loading from disk."
    #     )
    #     simulated_reads = output_path

    mapped_reads = map_reads_to_reference(
        reads=load_simulated_reads_from_disk(simulated_reads),
        reference=reference_genome,
    )

    return mapped_reads


def real_mapped_reads(
    bam_file: Union[Path, None] = None,
    reference_genome: Union[Path, None] = None,
):
    """
    Simulates reads and maps them to the reference genome.
    """

    mapped_reads = map_real_reads_to_reference(
        reads=load_real_reads_from_disk(bam_file),
        reference=reference_genome,
    )

    return mapped_reads
