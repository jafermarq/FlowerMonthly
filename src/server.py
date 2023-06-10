
from typing import Dict, Callable, Optional, Tuple


import torch
from omegaconf import DictConfig
from hydra.utils import instantiate
from flwr.common.typing import Scalar, NDArrays


from src.model_utils import  test, ndarrays_to_model


def gen_fit_config(fit_cfg: DictConfig):
    def fit_config(server_round: int) -> Dict[str, Scalar]:
        """Return a configuration with static batch size and (local) epochs."""
        return fit_cfg

    return fit_config


def get_evaluate_fn(
    testset,
    model_cfg: DictConfig,
) -> Callable[[NDArrays], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(
        server_round: int, parameters:NDArrays, config: Dict[str, Scalar], is_last_round: bool=False
    ) -> Optional[Tuple[float, float]]:
        """Use the entire CIFAR-10 test set for evaluation."""

        # determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Let's first instantiate the model
        model = instantiate(model_cfg)
        # Now set the model buffers with the parameters of the global model
        ndarrays_to_model(model, parameters)
        model.to(device)

        # here you could use the config to parameterise how the global evaluation is performed (e.g. use a particular bach size)
        # you could also use the `is_last_round` flag to switch between a global validation set and a global test set.
        # The global test set should be used only in the last round, while the global validation set can be used in all rounds.
        print(f"Is this the last round?: {is_last_round = }")

        testloader = torch.utils.data.DataLoader(testset, batch_size=128)

        # run global evaluation
        loss, accuracy = test(model, testloader, device=device)

        # Now you have evaluated the global model. This is the a good place to save a checkpoint if, for instance, a new
        # best global model is found (based on a global validation set).
        # If for instance you are using tensorboard to record global metrics or W&B (even better!!) this is the a good
        # place to log all the metrics you want.

        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate