import argparse
from helpers import initialize_pinecone, configs
from dna2vec.config_schema import DatasetConfigSchemaUniformSampling

parser = argparse.ArgumentParser(description="Config upsert")
parser.add_argument("--vector_db", nargs="+", default=["all"], help="Specify the reference chromosome used to populate the vector db")
parser.add_argument("--model", nargs="+", default=["trained-all-reg-mean-pooling"], help="Specify the model used to generate the embeddings")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--add_namespace", action="store_true", help="Add namespace to the upserted data")
parser.add_argument("--no_namespace", action="store_false", dest="add_namespace", help="Do not add namespace to the upserted data")
parser.add_argument("--pod_type", type=str, default="s1.x1")

def execute(checkpoint_queue: list, data_queue: list, device: str, add_namespace: bool, pod_type: str):
    store_generator = initialize_pinecone(checkpoint_queue, data_queue, device, pod_type)
    for store, data_alias, _ in store_generator:
        list_of_data_sources = []
        sources = data_alias.split(",")
        for source in sources:
            if source in configs["data_recipes"]:
                list_of_data_sources.append(configs["data_recipes"][source])
            else:
                list_of_data_sources.append(source)
        store.trigger_pinecone_upsertion(
            list_of_data_sources, add_namespace=add_namespace
        )


if __name__ == "__main__":
    import os

    args = parser.parse_args()
    data_queue = args.vector_db
    checkpoint_queue = args.model
    execute(checkpoint_queue, data_queue, args.device, args.add_namespace, args.pod_type)
