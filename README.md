# FlowerMonthly


The FlowerMonthy is a monthly online event organised by the team behind [Flower, A Friendly Federated Learning Framework](https://flower.dev/) that runs for one hour on the first Wednesday of each month (typically starting at 0900 SF, 1200 NY, 1700 LON, 1800 MAD, 2130 IST, 0000 北京) and is divide into four blocks of content:

  1. A platform update given by a member of the Flower team
  2. A 30 min presentation by a leading expert in Federated Learning
  3. A 15 min hands-on example of cool things people do with Flower
  4. Open discussion and Q&A

This repository contains some of the code examples presented in the Flower's FlowerMonthy series. You can see all past event [in the Flower site](https://flower.dev/conf/flower-monthly/). Jump on the fascinating FL train! 

> Join our [Slack channel](https://flower.dev/join-slack/) to chat directly to thousands already using Flower and to reach out to members of the Flower Team. Whether you are working on an amazing new feature or you hit a roadblock with your FL setup, [reach us also on GitHub](https://github.com/adap/flower) by submitting a PR or by opening an Issue.


## Content of this repo

> This repo will keep getting more examples after each Flower Monthly so be sure to come by & pull again.


To start this repo we have ported the [pytorch_simulation](https://github.com/adap/flower/tree/main/examples/simulation_pytorch) Flower example and adapted it so it works with [Hydra](https://hydra.cc/) configs to make the parameterisation of your FL experiments easy and flexible. The same could have been achieved using [AwesomeYaml](https://github.com/SamsungLabs/awesomeyaml) or other config systems. In fact, a previous version of this repo was entirely designed around AwesomeYaml (see tag `withAwesomeYaml` tag). I have added some small changes to the code provided by that example to make this repo more interesting, some of which is based on FlowerMonthly demos and talks. The code in this repo is validated using Flower's Virtual Client Engine for Simulation of FL workloads. However, the vast majority of the code here can be directly be used in gRPC-based Flower setups outside simulation.

The purpose of this repo is to showcase through simple examples different functionalies of [Flower](https://github.com/adap/flower) (**give it a :star: if you use it**) so you can later use it in your projects. With this in mind, the dataset considered here considered, its partitioning and the training protocol as a whole is kept fairly simple. Here I use CIFAR-10 and split it following [LDA](https://arxiv.org/abs/1909.06335) for a fixed value of \alpha (which you can tune in the configs). By default I generate a 100-client split and sample 10 clients per round (this is a simple but very typical _cross-silo_ FL setup)

Currently, this repo provides:

* A `configs/strategy_kd.yaml` config (based on 7 June 2023 FLowerSummit talk) showing how to do a simple form of federated Knowledge-distillation.
* A `configs/custom_strategy.yaml` config (based on 7 June 2023 FLowerSummit talk) showcasing how to design a custom Flower strategy with ease.
* A `configs/resnet18.yaml` config that changes the model being federated as well as the compute/memory resources allocated to each virtual client.
* A `configs/base.yaml` config with all the elements needed to define a complete FL setup. It uses a very lightweight model so all systems should be capable of running it (no GPU required).
* Integration with `AwesomeYaml`, so you can customise how your experiment runs directly from the command line.

## Setup

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


## Flower + Hydra for beginners

This section provides an introductory look into [`Hydra`](https://hydra.cc/) in the context of Federated Learning with Flower. You can run the code just as originally designed (no change in parameters) like this:
```bash
python main.py # this will use the default configs (i.e. everything defined in conf/base.yaml)
```

With Hydra you can easily change whatever you need from the config file without having to add a new `argparse` argument each time. For example:
```bash
python main.py server.client_per_round=5 # will use 5 clients per round instead of the default 10
python main.py dataset.lda_alpha=1000 # will use LDA alpha=1000 (making it IID) instead of the default value (1.0)

python main.py client.resources.num_cpus=4 # allocates 4 CPUs to each client (instead of the default 2 as defined in conf/client/cpu_client.yaml -- cpu_client is the default client to use as defined in conf/base.yaml->default.client)
```

In some settings you might want to make more substantial changes to the default config. For that, even though you couldl probably still doing from the command line, it can get messy... Instead, you can directly replace entire structures in your config with others. For example, let's say you want to change your entire client definition from the default one (check it in `conf/client/cpu_client.yaml`). You'll need to create a new yaml file, respecting the expected structure and place it at the same directory as `cpu_client.yaml`. This is exactly what I did with `gpu_client.yaml`. You can use the latter client as follows:
```bash
python main.py # will use the default `cpu_client.yaml`

# note that you'll need a GPU for this
python main.py client=gpu_client # will use the client as defined in `conf/client/gpu_client.yaml`
```

Let's say now that you have a concrete setting you'd like to evaluate often enough without having to modify from the "base" config to do so. The best way to do this would be to define a new top-level config with the `default:` your setup needs. For example, let's imagine you want your new setting to always use `resnet18.yaml`, `gpu_client.yaml` and `custom_strategy.yaml`. You can define a custom top-level config as follows:

```yaml
# this defines a top-level config (just like base.yaml) does but with changes to the `defaults` and the FL setup parameterised in `server:`

defaults: # i.e. configs used if you simply run the code as `python main.py`
  - client: gpu_client # this points to the file: client/cpu_client.yaml
  - model: resnet18 # this points to the file: model/resnet18.yaml
  - strategy: custom_strategy # points to strategy/custom_strategy.yaml

[...] # rest of the necessary elements: dataset, server, misc
```

The above config can be found in `conf/base_v2.yaml`, and you can execute as follows:

```bash
python main.py --config-name=base_v2 # this essentially overrides the config hardcoded in the @hydra decorator in the main.py
```