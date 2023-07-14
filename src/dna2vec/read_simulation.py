"""
Functions for simulating reads using ART as well as for reading in the simulated reads.
"""

import logging
import subprocess
from pathlib import Path
from typing import Literal

# biopython
from Bio import AlignIO, SeqIO

from dna2vec.utils import get_cache_dir, get_human_reference_genome

# import pysam

SEQUENCE_SYSTEMS = Literal[
    "GA1", "GA2", "HS10", "HS20", "HS25", "HSXn", "HSXt", "MinS", "MSv1", "MSv3", "NS50"
]


def simulate_reads(
    n_sequences: int,
    read_length: int,
    output: Path,
    reference_genome: Path | None = None,
    insertation_rate: float = 0.00009,
    deletion_rate: float = 0.00011,
    sequencing_system: SEQUENCE_SYSTEMS = "HS20",
):
    """
    Simulates unpaired reads using the ART command line interface.

    Args:
        n_sequences: Number of sequences to simulate.
        read_length: Length of the reads to simulate.
        reference_genome: Path to the reference genome to simulate reads from. If None it will download the human reference genome to the cache
            directory.
        output: Path to the output file. If None it will be saved to the cache directory.
        insertation_rate: Insertation rate for the simulation.
        deletion_rate: Deletion rate for the simulation.
        sequencing_system: Sequencing system to use for the simulation.
    """

    # art_illumina -ss HS20 -sam -i GRCh38_latest_genomic.fna -l 100 -c 1 -o chr2_read

    if reference_genome is None:
        reference_genome = get_human_reference_genome()

    logging.info(f"Simulating reads using ART")
    subprocess.run(
        [
            "art_illumina",
            "-ss",
            sequencing_system,
            "-sam",
            "-i",
            str(reference_genome),
            "-l",
            str(read_length),
            "-c",
            str(n_sequences),
            "-o",
            str(output),
            "insRate",
            str(insertation_rate),
            "delRate",
            str(deletion_rate),
        ]
    )

    return output


def read_simulated_reads(
    simulated_path: Path,
):
    """
    Read in the simulated reads using biopython.
    """
    # simulated_path = Path("/Users/au561649/Github/dna2vec/tests/data/chr2_read")

    logging.info(f"Reading simulated reads from {simulated_path}")

    fastq = simulated_path.with_suffix(".fq")

    # open using biopython
    records = SeqIO.parse(fastq, "fastq")
    records = list(records)
    record = records[0]
    dir(record)
    str(record.seq)
    record

