# FlowerMonthly


The FlowerMonthy is a monthly online event organised by the team behind [Flower, A Friendly Federated Learning Framework](https://flower.dev/) that runs for one hour on the first Wednesday of each month (typically starting at 0900 SF, 1200 NY, 1700 LON, 1800 MAD, 2130 IST, 0000 北京) and is divide into four blocks of content:

  1. A platform update given by a member of the Flower team
  2. A 30 min presentation by a leading expert in Federated Learning
  3. A 15 min hands-on example of cool things people do with Flower
  4. Open discussion and Q&A

This repository contains some of the code examples presented in the Flower's FlowerMonthy series. You can see all past event [in the Flower site](https://flower.dev/conf/flower-monthly/). Jump on the fascinating FL train! 

> Join our [Slack channel](https://flower.dev/join-slack/) to chat directly to thousands already using Flower and to reach out to members of the Flower Team. Whether you are working on an amazing new feature or you hit a roadblock with your FL setup, [reach us also on GitHub](https://github.com/adap/flower) by submitting a PR or by opening an Issue.


# Content of this repo

> This repo will keep getting more examples after each Flower Monthly so be sure to come by & pull again.


To start this repo we have ported the [pytorch_simulation](https://github.com/adap/flower/tree/main/examples/simulation_pytorch) Flower example and adapted it so it works with [AwesomeYaml](https://github.com/SamsungLabs/awesomeyaml) configs to make the parameterisation of your FL experiments easy and flexible. The same could have been achieved using [Hydra](https://hydra.cc/) or other config systems. I have added some small changes to the code provided by that example to make this repo more interesting, some of which is based on FlowerMonthly demos and talks. The code in this repo is validated using Flower's Virtual Client Engine for Simulation of FL workloads. However, the vast majority of the code here can be directly be used in gRPC-based Flower setups outside simulation.

Currently, this repo provides:

* A `configs/strategy_kd.yaml` config (based on 7 June 2023 FLowerSummit talk) showing how to do a simple form of federated Knowledge-distillation.
* A `configs/custom_strategy.yaml` config (based on 7 June 2023 FLowerSummit talk) showcasing how to design a custom Flower strategy with ease.
* A `configs/resnet18.yaml` config that changes the model being federated as well as the compute/memory resources allocated to each virtual client.
* A `configs/base.yaml` config with all the elements needed to define a complete FL setup. It uses a very lightweight model so all systems should be capable of running it (no GPU required).
* Integration with `AwesomeYaml`, so you can customise how your experiment runs directly from the command line.

### Setup

While there are different ways of setting up your Python environment, here I'll assume a [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installation is reachable from a standard bash/zsh terminal. These are the steps to setup the environment:

```bash
# create environment and activate
conda create -n flowermonthly python=3.8.13 -y
source activate flowermonthly

# install pytorch et al (you might want to adjust the command below depending on your platform/OS: https://pytorch.org/get-started/locally/)
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia -y

# install flower and other deps
pip install -r requirements.txt
```


## AwesomeYaml for beginners

This section provides an introductory look into [`AwesomeYaml`](https://github.com/SamsungLabs/awesomeyaml), an amazing tool to build yaml-like configs for Python environments. You can run the example just as originally designed (no change in parameters) like this:
```bash
python main.py configs/base.yaml
```

With AwesomeYaml you can easily change whatever you need from the config file without having to add a new `argparse` argument each time. For example:
```bash
python main.py configs/base.yaml server.client_per_round=5 # will use 5 clients per round instead of the default 10
python main.py configs/base.yaml dataset.lda_alpha=1000 # will use LDA alpha=1000 (making it IID) instead of the default value (1.0)

python main.py configs/base.yaml client.resources.num_gpus=0.5 client.optim.lr=0.1 # allocates 50% of a GPU to each client and changes their learning rate to 0.1
```

In some settings you might want to make more than a couple of changes to the `base.yaml` config. When this is the case, you can easily extend `configs/base.yaml` with a new config **only** including the changes you want. For example (this is `configs/resnet18.yaml`), let's say you want to change the model to a ResNet18 and change the resources for each client (this model is much bigger than the original model (`src.models.Net`) so probably you want clients to train it on GPU):

```yaml
---
!include configs/base.yaml
---

# here we extend the base config so the model is a ResNet18 adapted for CIFAR-10
# Because this is a much larger model, we also raise the resources Ray grants to each client

model:
  build: !bind:src.models.ResNet18
    num_classes: 10

client:
  resources:
    num_cpus: 4
    num_gpus: 0.5
```

To run the experiment with this config you'd simply do:

```bash
python main configs/resnet18.yaml

# you can still change settings on the fly (from any of the configs involved)
python main.py configs/resnet18.yaml server.num_rounds=50
```
