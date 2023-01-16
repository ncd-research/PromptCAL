# PromptCAL for Generalized Novel Category Discovery

- This repo contains codes for paper: [PromptCAL: Contrastive Affinity Learning via Auxiliary Prompts for Generalized Novel Category Discovery](https://arxiv.org/abs/2212.05590)

- ⭐️ **The original repository is [here](https://github.com/sheng-eatamath/PromptCAL).**


![main-5](./assets/main.png)

1. Download datasets in your folder and change their corresponding datapaths in `/config.py`, `/data/imagenet.py`, `/data/stanford_cars.py`, `/methods/contrastive_training/common.py`.
2. Before the training, create a `/cache` folder for saving checkpoints and results.
3. Before the evaluation, create a `/tmp` folder for saving intermediate and evaluation results.
4. The mode is run on a single A6000 GPU (48GB).

## Results

|                        | Stage 1               | Stage 2 |
| ---------------------- | --------------------- | ------- |
| CIFAR-10               |                       |         |
| CIFAR-100              | 77.08 / 80.75 / 69.75 |         |
| ImageNet-100           |                       |         |
| CUB (k_means_init=200) | 52.49 / 55.84 / 50.82 |         |
| SCars                  |                       |         |
| Aircraft               |                       |         |
| Herbarium19            |                       |         |

