
from typing import Dict, Callable, Optional, Tuple


import torch
import flwr as fl
from flwr.common.typing import Scalar, NDArrays
from awesomeyaml.config import Config

from src.model_utils import  test, ndarrays_to_model


def gen_fit_config(fit_cfg: Config):
    def fit_config(server_round: int) -> Dict[str, Scalar]:
        """Return a configuration with static batch size and (local) epochs."""
        return fit_cfg

    return fit_config


def get_evaluate_fn(
    testset,
    model_cfg: Config,
) -> Callable[[NDArrays], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(
        server_round: int, parameters:NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, float]]:
        """Use the entire CIFAR-10 test set for evaluation."""

        # determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = model_cfg.build()
        ndarrays_to_model(model, parameters)
        model.to(device)

        testloader = torch.utils.data.DataLoader(testset, batch_size=128)
        loss, accuracy = test(model, testloader, device=device)

        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate