"""
Functions for simulating reads using ART as well as for reading in the simulated reads.
"""

import logging
import subprocess
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

from collections import defaultdict
from Bio.SeqIO.FastaIO import FastaIterator
from Bio.Seq import Seq

from dna2vec.utils import get_cache_dir, download_human_reference_genome, load_human_reference_genome

import pysam

from pysam.libcalignedsegment import AlignedSegment


from dataclasses import dataclass

SEQUENCE_SYSTEMS = Literal[
    "GA1", "GA2", "HS10", "HS20", "HS25", "HSXn", "HSXt", "MinS", "MSv1", "MSv3", "NS50"
]


@dataclass
class ReadAndReference():
    """
    Class for storing a read and its reference.

    Attributes:
        read: The read.
        reference: The reference sequence as a string.
    """
    read: AlignedSegment
    reference: Union[str, None] = None

    def is_mapped(self) -> bool:
        return self.reference is not None

    def get_pair_as_string(self) -> Tuple[str, str]:
        """
        Get the pair as a tuple of strings.
        """
        return (self.read.query_sequence, self.reference)





def _format_scientific_notation(number: float) -> str:
    return "0" + format(9e-05, '.10f').strip("0")

def simulate_reads_to_disk(
    n_reads_pr_amplicon: int,
    read_length: int,
    output_path: Path,
    reference_genome: Union[Path, None] = None,
    insertion_rate: float = 0.00009,
    deletion_rate: float = 0.00011,
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

    fastq = simulated_path.with_suffix(".fq")
    sam = simulated_path.with_suffix(".sam")

    # load using pysam
    aligned_segments = pysam.AlignmentFile(str(sam))

    return list(aligned_segments)


def map_reads_to_reference(
        reads: List[AlignedSegment],
        reference: Optional[FastaIterator] = None,
):
    """
    Map a read to a reference genome.
    """
    if reference is None:
        reference = load_human_reference_genome()

    id2read = defaultdict(list)
    unmapped_reads = []
    for read in reads:
        _id = read.query_name.split("-")[0]
        
        unmapped_read = ReadAndReference(read=read)
        id2read[_id].append(unmapped_read)
        unmapped_reads.append(unmapped_read)
    
    for seq in reference:
        matches = id2read[seq.id]
        for match in matches:
            read = match.read
            start = read.reference_start
            length = read.query_length
            original_sequence = seq.seq[start:start+length]
            match.reference = str(original_sequence)
            assert read.query_sequence == read.seq

    return unmapped_reads


def simulate_mapped_reads(
    n_reads_pr_amplicon: int,
    read_length: int,
    insertion_rate: float = 0.00009,
    deletion_rate: float = 0.00011,
    sequencing_system: SEQUENCE_SYSTEMS = "HS20",
    reference_genome: Union[Path, None] = None,
):
    """
    Simulates reads and maps them to the reference genome.
    """

    output_path = get_cache_dir() / "simulated_reads" / f"reads_{n_reads_pr_amplicon}_{read_length}_{insertion_rate}_{deletion_rate}_{sequencing_system}"
    output_path.mkdir(parents=True, exist_ok=True)

    if not output_path.with_suffix(".sam").exists():
        simulated_reads = simulate_reads_to_disk(
            n_reads_pr_amplicon=n_reads_pr_amplicon,
            read_length=read_length,
            output_path=output_path,
            reference_genome=reference_genome,
            insertion_rate=insertion_rate,
            deletion_rate=deletion_rate,
            sequencing_system=sequencing_system,
        )
    else:
        logging.info(f"Simulated reads already exists at {output_path}. Loading from disk.")
        simulated_reads = output_path

    mapped_reads = map_reads_to_reference(
        reads=load_simulated_reads_from_disk(simulated_reads),
    )

    return mapped_reads

