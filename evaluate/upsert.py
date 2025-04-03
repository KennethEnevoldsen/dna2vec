from helpers import initialize_pinecone
from omegaconf import DictConfig
import hydra

def process_and_upsert_data(checkpoint_queue: list, data_queue: list, device: str, add_namespace: bool, pod_type: str, data_recipes: dict):
    store_generator = initialize_pinecone(checkpoint_queue, data_queue, device, pod_type)
    for store, data_alias, _ in store_generator:
        list_of_data_sources = []
        sources = data_alias.split(",")
        for source in sources:
            if source in data_recipes:
                list_of_data_sources.append(data_recipes[source][0])
            else:
                list_of_data_sources.append(source)
        store.trigger_pinecone_upsertion(
            list_of_data_sources, add_namespace=add_namespace
        )
        
@hydra.main(config_path="configs", config_name="upsert_config.yaml")
def upsert(cfg: DictConfig):
    data_queue = cfg.vector_db
    checkpoint_queue = cfg.model
    process_and_upsert_data(checkpoint_queue, data_queue, cfg.device, cfg.add_namespace, cfg.pod_type, cfg.data_recipes)



if __name__ == "__main__":
    upsert()
