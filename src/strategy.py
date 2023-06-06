from random import random
from typing import Union, Tuple, List, Dict, Callable


from awesomeyaml.config import Config

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from tqdm import tqdm

from flwr.server.strategy import FedAvg
from flwr.common.typing import Parameters, FitIns, FitRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.common import parameters_to_ndarrays

from .datasets import cifar10Transformation


class CustomFedAvg(FedAvg):
    """My customised FedAvg Strategy. It inherites from FedAvg.
    The ideas implemented here are designed with FL simulation in mind
    and for test research ideas including, but not limited to, understanding
    how a new strategy would behave in scenarios with unsual client participation
    patterns (and thefore requiring special sampling); simulating client failure (
    and therefore excluding certain updates from being aggregated).
    """
    def __init__(self, num_rounds: int, eval_every_n: int=5,
                 keep_ratio: float=0.5, drop_ratio: float=0.25, *args, **kwargs):

        self.num_rounds = num_rounds # total rounds
        self.eval_every_n = eval_every_n # global eval freq
        self.keep_ratio = keep_ratio # ratio of clients to resample in the following round
        self.client_update_drop = drop_ratio # ratio of client updates to discard from aggragation
        super().__init__(*args, **kwargs)

    def evaluate(self, server_round: int, parameters: Parameters):
        """Evaluates global model every N rounds. Last round is always
        considered and flagged as such (e.g. to use global test set)"""
        
        is_last_round = server_round == self.num_rounds
    
        if (server_round % self.eval_every_n == 0) or \
           (server_round == self.num_rounds):
            parameters_ndarrays = parameters_to_ndarrays(parameters)
            loss, metrics = self.evaluate_fn(server_round,
                                             parameters_ndarrays,
                                             config={},
                                             is_last_round=is_last_round)
            return loss, metrics
        else:
            print(f"Only evaluating every {self.eval_every_n} rounds...")
            return None

    def configure_fit(self, server_round: int, parameters: Parameters,
                      client_manager: ClientManager):
        """Configure the next round of training. In the first round we sample
        N clients from the M available and track which clients have been sampled.
        In subsequent rounds we sample again 100*keep_ratio % of the previously sampled
        clients and sample the remeaining (so we have N participants) out of the remaining
        ones. This is stochastic (it is likely not the same number of clients will always
        be kept for the next round)"""

        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        
        # construct instructions to be send to each client.
        # parameters: list of numpy arrays with model parameters
        # config: a python dictionary that parameterises the client's fit() method
        fit_ins = FitIns(parameters, config)

        # interface with the client manager to get statitics of the available pool of
        # clients that can be sampled in this given round.
        av_clients = client_manager.num_available()
        sample_size, min_num_clients = self.num_fit_clients(av_clients)
            
        if server_round == 1: # first round, random unifrom sampling (standard)

            clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients)

        else:
            # stochastically drop clients used in previous round
            clients = [cli for cli in self.prev_clients if random() < self.keep_ratio]
            print(f"Round {server_round} will resample clients: {[client.cid for client in clients]}")
            
            # sample more clients
            extra_clients = client_manager.sample(
                                num_clients=sample_size - len(clients),
                                min_num_clients=min_num_clients
                                )
            # append
            clients.extend(extra_clients)
            
        # record client proxies
        self.prev_clients = clients

        print(f"Roudn {server_round} sampled clients with cid: {[client.cid for client in self.prev_clients]}")

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(self, server_round: int,
                      results: List[Tuple[ClientProxy, FitRes]],
                      failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]):
        """Here we aggregate the results received from the clients and use them
        to update the global model state. Contrary to vanilla FedAvg, we stochastically
        drop a portion client_update_drop of the client updates received, then we aggregate
        using the ones kept. This could be used to simulate a number of scenarios, for example:
        errors in the communication channel when the clients send their updates to the server; 
        or simulate client failure or disconnection. (note that this same behaviour could be
        instead implemented in the client's fit() method with the additional benefit of not
        requiring to do training if we knew it that particular client was meant to be dropped anyway)."""

        # We can iterate over the results received from the clients
        # and record each client's id
        client_ids = [int(client_prox.cid) for client_prox, _ in results]

        # Create a drop mask
        drop_mask = [random() < (1.0 - self.client_update_drop) for _ in range(len(results))]
        dropped_cids = [cid for i, cid in enumerate(client_ids) if drop_mask[i]]
        print(f"CIDs of clients dropped: {dropped_cids}")

        # drop results according to mask
        results = [res for i, res in enumerate(results) if drop_mask[i]]

        # call the parent `aggregate_fit()` (i.e. that in standard FedAvg)
        return super().aggregate_fit(server_round, results, failures)


class CustomFedAvgWithKD(FedAvg):
    """My customised FedAvg Strategy for a simple setup with client-side distillation.
    The student model is the one federated as usual. The teacher model is first trained
    upon strategy creation and then sent to the client in each round (see `configure_fit`).
    The client uses the teacher to train the student locally using KD."""
    def __init__(self, num_rounds: int, teacher, kd_config, *args, **kwargs):

        self.num_rounds = num_rounds
        self.teacher_cfg = teacher # we store the callbale that can instantiate the teacher. We'll be sending this to the clients (in addition to the teacher weights)
        self.teacher = teacher.build() # instantiate teacher
        self.kd_config = kd_config

        # pre-train the teacher (for the purpose of this example we'll just use a handful of batches
        # using the training set). This will make the teach inmediately better than the the student
        # in the early stages of FL trianing (hence serving for our simple KD demo). Please note that
        # you'd normally will be doing the KD on a disjoint partition of data from that that's federated.
        # Likely this data would be from a common data distribution, so the KD is aligned.
        self._unrealistically_but_effectively_pretrain_the_teacher()

        # no need to do anything else, call default behaviour from parent vanilla FedAvg
        super().__init__(*args, **kwargs)


    def _unrealistically_but_effectively_pretrain_the_teacher(self, path_to_data="./data"):

        # Do training as you'd normally do in a centralised setup
        train_set = CIFAR10(root=path_to_data, train=True, download=True, transform=cifar10Transformation())

        trainloader = DataLoader(train_set, batch_size=self.kd_config.teacher_pretrain.batch_size, num_workers=4)
        optim = self.kd_config.teacher_pretrain.optim(self.teacher.parameters())

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        criterion = torch.nn.CrossEntropyLoss()
        self.teacher.to(device)
        self.teacher.train()

        print(f"Pretraining teacher for {self.kd_config.teacher_pretrain.num_batches} batches of size {self.kd_config.teacher_pretrain.batch_size}")
        with tqdm(total=len(train_set), desc=f'pseudo-pretraining teacher') as t:
            for i, (images, labels) in enumerate(trainloader):
                images, labels = images.to(device), labels.to(device)
                optim.zero_grad()
                loss = criterion(self.teacher(images), labels)
                loss.backward()
                optim.step()

                t.update(images.shape[0])

                if i + 1 == self.kd_config.teacher_pretrain.num_batches:
                    break
        print("Teacher is pretrained")

    def _get_teacher_asarray(self):
        self.teacher.cpu()
        return [val.numpy() for _, val in self.teacher.state_dict().items()]

    def configure_fit(self, server_round: int, parameters: Parameters,
                      client_manager: ClientManager):
        """Configure the next round of training. Standard behaviour as in FedAvg
        but fit instructions have been extended to include the teacher model and
        the config that describes how to do KD on the client side. """

        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)

        # flatten and add teacher model and KD config to config dict
        # this config will be received by each participating client
        config['teacher_cfg'] = self.teacher_cfg
        config['teacher_arrays'] = self._get_teacher_asarray()
        config['KD_config'] = self.kd_config.student_train

        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def evaluate(self, server_round: int, parameters: Parameters):
        """Evaluates global model. Flags if last round in FL training. Recall
        that `parameters` are the weights of the model being federated (i.e. 
        the student in this KD example)"""
        
        is_last_round = server_round == self.num_rounds
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        loss, metrics = self.evaluate_fn(server_round,
                                            parameters_ndarrays,
                                            config={},
                                            is_last_round=is_last_round)
        return loss, metrics
