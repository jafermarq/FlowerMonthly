from time import time
from typing import Dict
from pathlib import Path
from collections import OrderedDict

import torch
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

import flwr as fl
from flwr.common.typing import Scalar

from .datasets import get_dataloader
from .model_utils import train, train_with_kd, test, model_as_ndarrays, ndarrays_to_model

class FlowerClient(fl.client.NumPyClient):
    """A very standard Flower client customisable via AwesomeYAML configs.
    Simple but covers 95%+ of what you'd want to do in FL."""
    def __init__(self, cid: str, fed_dir_data: str, cfg: DictConfig):
        self.cid = cid
        self.fed_dir = Path(fed_dir_data)
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}

        self.cfg = cfg

        # Instantiate model (because the client class might have also been instantiated via Hydra, you want to make sure that
        # the client was instantiated with _recursive_=False. Else the below will fail. Even worse! all clients will be pointing
        # to the same object so it will definetively create problems -- which can be solved via copy.deepcopy() but why doing it that way?)
        self.net = instantiate(self.cfg.model)

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

        optimizer = instantiate(self.cfg.optim, params=self.net.parameters())
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


class FlowerClientWithKD(FlowerClient):
    """A Flower client that behaves as the standard client above for the most part.
    The main exception being that local training is done using Knowledge-Distillation
    using as teacher a model sent from the server. Under this formulation of federated
    KD, the server sends two models to the clients: a pre-trained teacher and a student.
    The latter is the one being updated/trained by the clients and hence the one being
    aggregated by the strategy in the server. Please note this is a very simple setup
    for demonstration purposes."""


    def _instantiate_teacher(self, teacher: DictConfig, teacher_arrays):
        teacher_model = instantiate(teacher) # instantiate
        
        # copy params sent by server
        params_dict = zip(teacher_model.state_dict().keys(), teacher_arrays)
        state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
        teacher_model.load_state_dict(state_dict, strict=True)
        return teacher_model

    def fit(self, parameters, config):
        """The fit() method receives `parameters`, i.e., the parameters of the model being
        federated. In our example this correspond to the student model. The teacher model
        is sent as part of the config. Before we can train using KD, we need to instantiate
        the teacher."""

        # update the local model with the parameters sent by the server.
        self.set_parameters(parameters)

        # instantiate teacher with parameters sent from server
        # (We could of course instantiate the teacher in the constructor of this class and only
        # update it's weights here -- for example if we have a more elaborated setup where the
        # teacher is also being periodically updated by the server)
        teacher = self._instantiate_teacher(config["teacher_cfg"], config["teacher_arrays"])
        print(f"Client {self.cid} has loaded teacher network successfully!")

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

        # Send the teacher to device too (be mindful that this will increase the resource utilisation of the clients.
        # this will likely require you to revisit the `client_resources` you setup when launching the simulation. This
        # is critical if you plan to use a high capacity teacher)
        teacher.to(self.device)

        # track parameters of student network
        optimizer = instantiate(self.cfg.optim, params=self.net.parameters())

        # Train with distillation, We time it
        start_t = time()
        train_with_kd(self.net, teacher, config["KD_config"], trainloader, epochs=config["epochs"], device=self.device, optim=optimizer)
        # time (in secods) that took to do run `train_with_kd`
        total_t = time() - start_t

        # Return local model and statistics. You can return whatever you want using the last argument (the "Metrics", as are called in Flower)
        # Using Metrics is great to track in the sever different info about how the training on the clients is going
        # or when you are experiment with new setups. Just be mindful that, to stay true the FL spirit, no sensible info
        # should be sent back to the server. Even in simulation settings, incorporating client-side info that would normally
        # not be available in real deployments might limit the effectiveness of the method you are investigated when deploying
        # it out in the wild. 
        return self.get_parameters(config), len(trainloader.dataset), {"fit_time": total_t}
