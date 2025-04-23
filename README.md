# ORW-CFM-W2: Online Reward-Weighted Conditional Flow Matching with Wasserstein-2 Regularization

This repository contains the implementation of "Online Reward-Weighted Fine-Tuning of Flow Matching with Wasserstein Regularization," a method for fine-tuning flow-based generative models using reinforcement learning.

## Overview

ORW-CFM-W2 is a novel reinforcement learning approach for fine-tuning continuous flow-based generative models to align with arbitrary user-defined reward functions. Unlike previous methods that require filtered datasets or gradients of rewards, our method enables optimization with arbitrary reward functions while preventing policy collapse through Wasserstein-2 distance regularization.

## Method

Our approach integrates reinforcement learning into the flow matching framework through three key components:

1. **Online Reward-Weighting**: Guides the model to prioritize high-reward regions in the data manifold
2. **Wasserstein-2 Regularization**: Prevents policy collapse and maintains diversity
3. **Tractable W2 Distance Bound**: Enables efficient computation of the W2 distance in flow matching models

The loss function is defined as:

$$\mathcal{L}_{\text{ORW-CFM-W2}} = \mathbb{E}_{t \sim \mathcal{U}(0,1), x_1 \sim q(x_1; \theta_{\text{ft}}), x \sim p_t(x|x_1)}[w(x_1) \|v_{\theta_{\text{ft}}}(t, x) - u_t(x|x_1)\|^2 + \alpha \|v_{\theta_{\text{ft}}}(t, x) - v_{\theta_{\text{ref}}}(t, x)\|^2]$$

Where:
- $w(x_1) \propto r(x_1)$ is the weighting function proportional to the reward
- $v_{\theta_{\text{ft}}}$ is the fine-tuned model's vector field
- $v_{\theta_{\text{ref}}}$ is the reference (pre-trained) model's vector field
- $u_t(x|x_1)$ is the true conditional vector field
- $\alpha$ is the regularization coefficient that controls the trade-off between reward and diversity

## Implementation

The core implementation is in the `ORWCFMTrainer` class, which handles:

1. Initialization of models (network model, last policy, reference model)
2. Sampling from the current policy
3. Computing rewards for samples
4. Computing the loss with both FM and W2 components
5. Updating the model parameters
6. Periodically updating the sampling policy

## Usage

### Basic Usage

```python
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from torchcfm.models.unet.unet import UNetModelWrapper
from orwcfm import ORWCFMTrainer

# Define configuration
config = {
    'learning_rate': 2e-4,
    'warmup_steps': 5000,
    'w2_coefficient': 1.0,  # alpha parameter for W2 regularization, we encourage you to use at least alpha>=1.0
    'temperature': 0.5,     # tau parameter for reward weighting
    'grad_clip': 1.0,
    'batch_size': 128,
    'text_prompts': ["An image of dog", "Not an image of dog"],
    'use_wandb': True,
    'wandb_project': 'flow-matching',
    'run_name': 'orw-cfm-w2',
    'savedir': './results',
    'ref_path': './pretrained/model.pt'  # Path to pre-trained model
}

# Initialize model
model = UNetModelWrapper(
    dim=(3, 32, 32),
    num_res_blocks=2,
    num_channels=128,
    channel_mult=[1, 2, 2, 2],
    num_heads=4,
    num_head_channels=64,
    attention_resolutions="16",
    dropout=0.1
)

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize trainer
trainer = ORWCFMTrainer(model, config, device)

# Load pre-trained model
trainer.load_pretrained(config['ref_path'])

# Train model
trainer.train(
    num_epochs=1000,
    steps_per_epoch=100
)

# Save checkpoint
trainer.save_checkpoint('./checkpoints/orw_cfm_w2.pt')
```

### Key Parameters

- **w2_coefficient (alpha)**: Controls the strength of the W2 regularization. Higher values prioritize staying close to the reference model, leading to more diverse outputs. Lower values prioritize reward maximization.
- **temperature (tau)**: Controls the sharpness of the reward weighting. Higher values lead to more aggressive focusing on high-reward regions.

## Theoretical Guarantees

Our method provides the following theoretical guarantees:

1. **Convergence Behavior**: The data distribution after N epochs evolves according to:
   
   $$q^N_{\theta}(x_1) \propto w(x_1) q^{N-1}_{\theta}(x_1) \exp(-\beta D^{N-1}(x_1))$$

   Where $D^{N-1}(x_1)$ measures the discrepancy between the current and reference models.

2. **Limiting Behavior**: Without regularization (α=0), the model converges to a delta distribution centered at the maximum reward point.

3. **Reward-Diversity Trade-off**: W2 regularization enables a controllable trade-off between reward maximization and diversity preservation.

## Citation

If you find this code useful for your research, please consider citing our paper:

```bibtex
@inproceedings{
fan2025online,
title={Online Reward-Weighted Fine-Tuning of Flow Matching with Wasserstein Regularization},
author={Jiajun Fan and Shuaike Shen and Chaoran Cheng and Yuxin Chen and Chumeng Liang and Ge Liu},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=2IoFFexvuw}
}
```

## Dependencies

- PyTorch
- TorchCFM
- wandb (optional, for logging)
- tqdm

## License

[MIT License](https://mit-license.org/)