# FlowerMonthly


Here I have ported the [pytorch_simulation](https://github.com/adap/flower/tree/main/examples/simulation_pytorch) Flower example and adapted it so it works with [awesomeyaml](https://github.com/SamsungLabs/awesomeyaml) configs to make things easy and flexible. I have added some small changes to the code provided by that example to make this repo more interesting.

### Setup

```bash

# create environment and activate
conda create -n flowermonthly python=3.8.13 -y
source activate flowermonthly

# install pytorch et al (you might want to adjust the command below depending on your platform/OS: https://pytorch.org/get-started/locally/)
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia -y

# install flower and other deps
pip install -r requirements.txt
```


## AwesomeYaml

Run the example just as originally designed (no change in parameters):
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
