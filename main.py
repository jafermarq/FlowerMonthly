import pickle
from pathlib import Path

import flwr as fl
import hydra
from hydra.utils import call, instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


from src.server import get_evaluate_fn
from src.strategy import CustomFedAvgWithModelSaving


@hydra.main(version_base=None, config_path="conf", config_name="base")
def run(cfg : DictConfig):

    print(OmegaConf.to_yaml(cfg))

    # Each time you run this, Hydra will create a new directory containing
    # the config you used as well as the generated log. You can retrieve
    # the path to this directory as shown below. Ideally, here is where
    # you'd be saving any output (e.g. checkpoints) for this experiment
    save_path = HydraConfig.get().runtime.output_dir
    print(f"Output directory for this experiment: {save_path}")

    # let's prepare the dataset (download + partition)
    fed_dir, testset = call(cfg.dataset.prepare)

    # let's define our strategy (instantiating the object defined in the config)
    # You can pass additional arguments needed for the object (that weren't possible
    # to define in the config maybe becasue they are defined at runtime). You need to
    # use keyword arguments.
    # in this case, the function to evaluate the global model requires passing the testset object
    # Our strategy config might contain other nodes with _target_. Often, we want to delay when these
    # are instantiated until, for instance, all variables needed to do so are ready. We set _recursive_=False
    # to leave those nodes un-initialised (we set that in the config itself with the appropiate value)
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
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.server.pool, # total number of clients in the experiment
        client_resources=cfg.client.resources, # resources that will be reserved for each client
        config=fl.server.ServerConfig(num_rounds=cfg.server.num_rounds),
        strategy=strategy,
        ray_init_args=ray_init_args,
    )

    # now you can for instance save your results in to a Python pickle
    extra_results = {} # add here any other results you want to save

    # if your strategy is keepting track of some variables you want to retreive once
    # the experiments is completed, you can totally do so. You might want to do this
    # for instance, in order to save the global model weights
    if isinstance(strategy, CustomFedAvgWithModelSaving):
        model_parameters = strategy.global_parameters
        extra_results['global_parameters'] = model_parameters

    # add everything into a single dictionary
    data = {'history': history, **extra_results}

    results_path = Path(save_path)/'results.pkl'
    # save to pickle
    with open(str(results_path), "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Results saved into: {results_path}")

if __name__ == "__main__":

    run()
