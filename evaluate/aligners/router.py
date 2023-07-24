from typing import Literal, Optional
from pathlib import Path

class Router:
    def __init__(self, 
                 mode: Literal["minimap", "smith waterman", "bwamem2", "bowtie2"], 
                 aligners_root_path: str,
                 index_path: Optional[str] = None,
    ):
        pass