import argparse
from helpers import initialize_pinecone, data_recipes
from dna2vec.config_schema import DatasetConfigSchemaUniformSampling

parser = argparse.ArgumentParser(description="Config upsert")
parser.add_argument("--recipes", type=str)
parser.add_argument("--checkpoints", type=str)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--add_namespace", type=bool, default=False)


def execute(checkpoint_queue: list, data_queue: list, device: str, add_namespace: bool):
    store_generator = initialize_pinecone(checkpoint_queue, data_queue, device)
    for store, data_alias, _ in store_generator:
        list_of_data_sources = []
        sources = data_alias.split(",")
        for source in sources:
            if source in data_recipes:
                list_of_data_sources.append(data_recipes[source])
            else:
                list_of_data_sources.append(source)
        store.trigger_pinecone_upsertion(
            list_of_data_sources, add_namespace=add_namespace
        )


if __name__ == "__main__":
    import os

    args = parser.parse_args()
    data_queue = args.recipes.split(";")
    checkpoint_queue = args.checkpoints.split(";")
    execute(checkpoint_queue, data_queue, args.device, args.add_namespace)
