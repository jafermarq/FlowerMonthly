import flwr as fl
import hydra
from hydra.utils import call, instantiate
from omegaconf import DictConfig, OmegaConf


from src.server import get_evaluate_fn


@hydra.main(version_base=None, config_path="conf", config_name="base")
def run(cfg : DictConfig):

    print(OmegaConf.to_yaml(cfg))

    # let's prepare the dataset (download + partition)
    fed_dir, testset = call(cfg.dataset.prepare)

    # let's define our strategy (instantiating the object defined in the config)
    # You can pass additional arguments needed for the object (that weren't possible
    # to define in the config maybe becasue they are defined at runtime). You need to
    # use keyword arguments.
    # in this case, the function to evaluate the global model requires passing the testset object
    strategy = instantiate(cfg.strategy, evaluate_fn=get_evaluate_fn(testset, cfg.model))
    
    def client_fn(cid: str):
        # Create a single client instance
        # The type of client class is set at runtime based on the config used. Recall we need to pass
        # extra arguemtns that weren't available when when the config is parsed. Also, let's not instantiate
        # every object inside the client config (use `_recursive_`=False). This will give us full control on
        # when instantiation happens.
        return instantiate(cfg.client.object, cid=cid, fed_dir_data=fed_dir, _recursive_=False)
    
    # (optional) specify Ray config
    # If you want to do multi-node simulations you want the VCE to attach to an existing Ray server
    ray_init_args = {"include_dashboard": False, "address": "auto" if cfg.misc.attach else None}

    # start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.server.pool, # total number of clients in the experiment
        client_resources=cfg.client.resources, # resources that will be reserved for each client
        config=fl.server.ServerConfig(num_rounds=cfg.server.num_rounds),
        strategy=strategy,
        ray_init_args=ray_init_args,
    )

if __name__ == "__main__":

    run()
