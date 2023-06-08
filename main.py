import argparse
import flwr as fl
import awesomeyaml


from src.server import get_evaluate_fn


parser = argparse.ArgumentParser(description="StreamFL")
parser.add_argument('yamls', nargs='+', type=str)
parser.add_argument('--attach', action='store_true', help='Set it if you want to attach to a Ray Server')



if __name__ == "__main__":

    # parse config
    args = parser.parse_args()
    cfg = awesomeyaml.Config.build_from_cmdline(*args.yamls)

    fed_dir, testset = cfg.dataset.prepare

    # configure the strategy
    # The Strategy class is set at runtime based on the config you use
    strategy = cfg.server.strategy(evaluate_fn=get_evaluate_fn(testset, cfg.model))

    def client_fn(cid: str):
        # Create a single client instance
        # The type of client class is set at runtime based on the config used
        return cfg.client.build(cid, fed_dir)

    # (optional) specify Ray config
    # If you want to do multi-node simulations you want the VCE to attach to an existing Ray server
    ray_init_args = {"include_dashboard": False, "address": "auto" if args.attach else None}

    # start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.server.pool, # total number of clients in the experiment
        client_resources=cfg.client.resources, # resources that will be reserved for each client
        config=fl.server.ServerConfig(num_rounds=cfg.server.num_rounds),
        strategy=strategy,
        ray_init_args=ray_init_args,
    )