from typing import Dict, List

import numpy as np


class VectorStore:
    seq2vec: Dict[str, np.ndarray] = {}
    device: str

    def __init__(
        self,
        sequences: List[str] = [],
        device: str = "cuda:4",
    ) -> None:
        self.device = device
        self.add_sequences(sequences)

    def clear_scores(self) -> None:
        self.seq2vec = {}

    def add_sequences(self, sequences: List[str]) -> None:
        for sent, emb in zip(sequences, self.embed(sequences)):
            self.seq2vec[sent] = emb

    def embed(self, sequences: List[str]) -> List[np.ndarray]:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(sequences)
        return embeddings  # type: ignore

    def cosine_similarity(self, seq1: str, seq2: str) -> float:
        from scipy.spatial.distance import cosine

        if seq1 not in self.seq2vec:
            self.seq2vec[seq1] = self.embed([seq1])[0]
        if seq2 not in self.seq2vec:
            self.seq2vec[seq2] = self.embed([seq2])[0]
        return 1 - cosine(self.seq2vec[seq1], self.seq2vec[seq2])  # type: ignore

    def visualize(self):
        """
        Visualize the vectors in 2D space using UMAP and bokeh
        """
        import umap
        from bokeh.io import output_notebook, show
        from bokeh.plotting import figure

        # create 2 dimensional embedding
        umap_embeddings = umap.UMAP(
            n_neighbors=15, n_components=2, metric="cosine"
        ).fit_transform(list(self.seq2vec.values()))

        # plot the 2 dimensional data points
        plot = figure(
            title="UMAP projection of the DNA sequences",
            plot_width=600,
            plot_height=600,
        )
        plot.scatter(
            x=umap_embeddings[:, 0],
            y=umap_embeddings[:, 1],
            marker="circle",
            line_color="navy",
            fill_color="orange",
            alpha=0.5,
        )

        show(plot)
