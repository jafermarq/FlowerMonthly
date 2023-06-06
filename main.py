import argparse
from typing import Dict, Callable, Optional, Tuple


import torch
import flwr as fl
from flwr.common.typing import Scalar
import awesomeyaml


from src.client import FlowerClient
from src.server import get_evaluate_fn


parser = argparse.ArgumentParser(description="StreamFL")
parser.add_argument('yamls', nargs='+', type=str)
parser.add_argument('--attach', action='store_true')



if __name__ == "__main__":

    # parse config
    args = parser.parse_args()
    cfg = awesomeyaml.Config.build_from_cmdline(*args.yamls)

    fed_dir, testset = cfg.dataset.prepare

    # configure the strategy
    strategy = cfg.server.strategy(evaluate_fn=get_evaluate_fn(testset, cfg.model))

    def client_fn(cid: str):
        # create a single client instance
        return cfg.client.build(cid, fed_dir)

    # (optional) specify Ray config
    ray_init_args = {"include_dashboard": False, "address": "auto" if args.attach else None}

    # start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.server.pool,
        client_resources=cfg.client.resources,
        config=fl.server.ServerConfig(num_rounds=cfg.server.num_rounds),
        strategy=strategy,
        ray_init_args=ray_init_args,
    )