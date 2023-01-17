# PromptCAL for Generalized Novel Category Discovery

- This repo contains codes for paper: [PromptCAL: Contrastive Affinity Learning via Auxiliary Prompts for Generalized Novel Category Discovery](https://arxiv.org/abs/2212.05590)

- ⭐️ **The original repository is [here](https://github.com/sheng-eatamath/PromptCAL).**


![main-5](./assets/main.png)

1. Download datasets in your folder and change their corresponding datapaths in `/config.py`, `/data/imagenet.py`, `/data/stanford_cars.py`, `/methods/contrastive_training/common.py`.
2. Before the training, create a `/cache` folder for saving checkpoints and results.
3. Before the evaluation, create a `/tmp` folder for saving intermediate and evaluation results.
4. The mode is run on a single A6000 GPU (48GB).

## Results

|              | Stage 1               | Stage 2               |
|--------------|-----------------------|-----------------------|
| CIFAR-10     |              TBU      |     TBU               |
| CIFAR-100    | 77.08 / 80.75 / 69.75 | 81.76 / 84.94 / 75.41 |
| ImageNet-100 |                  TBU  |       TBU             |
| CUB          | 52.96 / 58.57 / 50.15 | 63.90 / 67.04 / 62.33 |
| SCars        | 42.97 / 64.77 / 32.44 | 51.40 / 70.26 / 42.29 |
| Aircraft     | TBU                   |          TBU          |
| Herbarium19  |     TBU               |       TBU             |

