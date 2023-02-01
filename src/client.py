from pathlib import Path
from typing import Dict

import torch
import flwr as fl
from flwr.common.typing import Scalar
from awesomeyaml.config import Config

from .datasets import get_dataloader
from .model_utils import train, test, model_as_ndarrays, ndarrays_to_model

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: str, fed_dir_data: str, cfg: Config):
        self.cid = cid
        self.fed_dir = Path(fed_dir_data)
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}

        self.cfg = cfg

        # Instantiate model
        self.net = self.cfg.model.build()

        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_parameters(self, config):
        return model_as_ndarrays(self.net)

    def set_parameters(self, parameters):
        ndarrays_to_model(self.net, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        # Load data for this client and get trainloader
        trainloader = get_dataloader(
            self.fed_dir,
            self.cid,
            is_train=True,
            batch_size=config["batch_size"],
            workers=config["num_cpu"],
        )

        # Send model to device
        self.net.to(self.device)

        optimizer = self.cfg.optim(self.net.parameters())
        # Train
        train(self.net, trainloader, epochs=config["epochs"], device=self.device, optim=optimizer)

        # Return local model and statistics
        return self.get_parameters(config), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        # Load data for this client and get trainloader
        valloader = get_dataloader(
            self.fed_dir, self.cid, is_train=False, batch_size=50, workers=2
        )

        # Send model to device
        self.net.to(self.device)

        # Evaluate
        loss, accuracy = test(self.net, valloader, device=self.device)

        # Return statistics
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}
