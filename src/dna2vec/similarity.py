from typing import Dict, List, Literal

import pandas as pd
from pydantic import BaseConfig, BaseModel


class VectorStore(BaseModel):

    dataframe: pd.DataFrame
    delimiter: str
    scores: Dict[str, float]
    device: str

    class Config(BaseConfig):
        arbitrary_types_allowed = True

    def __init__(
        self,
        df,
        delimiter=" -> ",
        device: str = "cuda:4",
        ABORT_LIMIT: int = 5,
        SPARSITY_LIMIT: int = 1000,
    ) -> None:

        super(VectorStore, self).__init__(
            dataframe=df, delimiter=delimiter, scores={}, device=device
        )

        if len(df) < ABORT_LIMIT:  # too few rows; perhaps not enough data points
            raise ValueError("Dataset is of small size. Aborting...")
        if len(df) < SPARSITY_LIMIT:
            print("You are attempting a vector store on data that is sparse.")

        required_columns = [
            "Total_Relationships",
            "Total_Relationships+Template",
            "Argument1+Template",
            "Argument2+Template",
        ]
        for col in required_columns:
            if col not in self.dataframe.columns:
                raise ValueError(f"{col} column is missing from dataframe.")

    def create_local_optimized_vector_store(
        self,
        mode: Literal[
            "sentence_transformer", "sentence_transformer_with_role_templates"
        ] = "sentence_transformer",
        phrase_mode: Literal["arguments", "relationships"] = "arguments",
    ) -> None:

        if mode == "sentence_transformer":
            self.run(mode=phrase_mode, template_assistance=False)
        elif mode == "sentence_transformer_with_role_templates":
            self.run(mode=phrase_mode, template_assistance=True)

    def clear_scores(self) -> None:
        self.scores = {}

    def get_scores(self, phrases: List[str]) -> None:
        from sentence_transformers import SentenceTransformer, util

        model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
        # device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        # print(device)
        # model.to(device)

        # TODO: check these parameters
        paraphrases = util.paraphrase_mining(
            model,
            phrases,
            show_progress_bar=True,
            batch_size=32,
            max_pairs=20000,
            corpus_chunk_size=5000,
            query_chunk_size=1000,
        )
        return paraphrases

    def run(
        self,
        mode: Literal["arguments", "relationships"] = "arguments",
        template_assistance: Literal[True, False] = True,
    ) -> None:

        if self.scores:
            raise Exception(
                "Recompiling previously compiled scores. Please explicitly call: clear_scores()"
            )

        if mode == "arguments":

            if not template_assistance:
                arg1s = self.dataframe["Argument1"].to_list()
                arg2s = self.dataframe["Argument2"].to_list()
                stopper = len(arg1s)
                paraphrases = self.get_scores(arg1s + arg2s)  # watch for top-K
            else:
                arg1s = self.dataframe["Argument1+Template"].to_list()
                arg2s = self.dataframe["Argument2+Template"].to_list()
                stopper = len(arg1s)
                paraphrases = self.get_scores(arg1s + arg2s)  # watch for top-K

            # Store optimally
            for paraphrase in paraphrases:

                score, i, j = paraphrase
                first_phrase: str
                second_phrase: str

                if i >= stopper:
                    first_phrase = arg2s[i - stopper]
                else:
                    first_phrase = arg1s[i]

                if j >= stopper:
                    second_phrase = arg2s[j - stopper]
                else:
                    second_phrase = arg1s[j]

                # write it both ways - this is a symmetric operation
                self.scores[
                    self.delimiter.join([first_phrase, second_phrase])
                ] = score  # optimized for speed
                self.scores[
                    self.delimiter.join([second_phrase, first_phrase])
                ] = score  # optimized for speed

                # you are not guaranteed to get all pairs here. See:
                # https://www.sbert.net/examples/applications/paraphrase-mining/README.html

        elif mode == "relationships":

            if not template_assistance:
                total_relationships = self.dataframe["Total_Relationships"].to_list()
                paraphrases = self.get_scores(total_relationships)  # watch for top-K
            else:
                total_relationships = self.dataframe[
                    "Total_Relationships+Template"
                ].to_list()
                paraphrases = self.get_scores(total_relationships)  # watch for top-K

            # Store optimally
            for paraphrase in paraphrases:
                score, i, j = paraphrase
                self.scores[
                    self.delimiter.join(
                        [total_relationships[i], total_relationships[j]]
                    )
                ] = score
                self.scores[
                    self.delimiter.join(
                        [total_relationships[j], total_relationships[i]]
                    )
                ] = score
                # optimized for speed

    def get_score(self, phrase1: str, phrase2: str) -> float:
        try:
            return self.scores[self.delimiter.join([phrase1, phrase2])]
        except KeyError:
            return -1  # works for cosine
