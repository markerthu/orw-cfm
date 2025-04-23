import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import wandb
from tqdm import tqdm
import copy
import os
from typing import List, Tuple

from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from torchcfm.models.unet.unet import UNetModelWrapper
from utils_cifar import generate_samples
from clip_reward import get_text_image_clip_score


class ORWCFMTrainer:
    def __init__(
            self,
            model: nn.Module,
            config: dict,
            device: torch.device
    ):
        self.device = device
        self.config = config

        # Initialize models
        self.net_model = model.to(device)
        self.last_policy = copy.deepcopy(model).to(device)
        self.ref_model = copy.deepcopy(model).to(device)

        # Initialize Flow Matcher
        self.fm = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)

        # Initialize optimizer and scheduler
        self.optimizer = Adam(self.net_model.parameters(), lr=config['learning_rate'])
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(step, config['warmup_steps']) / config['warmup_steps']
        )

        # Training parameters
        self.alpha = config['w2_coefficient']
        self.beta = config['temperature']
        self.grad_clip = config['grad_clip']
        self.use_wandb = config.get('use_wandb', False)
        self.parallel = config.get('parallel', False)
        self.savedir = config.get('savedir', './results')

        # Initialize wandb if needed
        if self.use_wandb:
            wandb.init(
                project=config['wandb_project'],
                name=config['run_name']
            )

    def load_pretrained(self, path: str):
        """Load pretrained checkpoints"""
        checkpoint = torch.load(path)
        self.net_model.load_state_dict(checkpoint['net_model'])
        self.last_policy.load_state_dict(checkpoint['net_model'])
        self.ref_model.load_state_dict(checkpoint['net_model'])

        if 'optim' in checkpoint and 'sched' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optim'])
            self.scheduler.load_state_dict(checkpoint['sched'])

        # Set models to appropriate modes
        self.ref_model.eval()
        self.last_policy.eval()

    @torch.no_grad()
    def sample_batch(self, ep: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample batch using last policy and compute rewards"""
        self.last_policy.eval()

        # Generate samples using last policy
        _, samples = generate_samples(
            self.last_policy,
            self.parallel,
            self.savedir,
            ep,
            net_="last_policy",
            save_img=False,
            use_wandb=self.use_wandb,
            log_image_interval=50,
            return_x0=True
        )

        # Compute rewards using CLIP
        image_prob, _ = get_text_image_clip_score(
            image=samples,
            text=self.config['text_prompts'],
            return_logit=True
        )
        rewards = image_prob[:, 0] - image_prob[:, 1] # you can use other rewards

        return samples, rewards

    def compute_loss(
            self,
            x1: torch.Tensor,
            weights: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Compute ORW-CFM-W2 loss"""
        # Generate noise
        x0 = torch.randn_like(x1)

        # Get flow matching components
        t, xt, ut = self.fm.sample_location_and_conditional_flow(x0, x1)

        # Compute vector fields
        vt = self.net_model(t, xt)
        vt_ref = self.ref_model(t, xt).detach()

        # Compute losses
        fm_loss = ((vt - ut) ** 2).mean(dim=(1, 2, 3))
        w2_loss = ((vt - vt_ref) ** 2).mean(dim=(1, 2, 3))

        # Combine losses
        total_loss = torch.mean(weights * fm_loss + self.alpha * w2_loss)

        metrics = {
            'fm_loss': fm_loss.mean().item(),
            'w2_loss': w2_loss.mean().item(),
            'total_loss': total_loss.item()
        }

        return total_loss, metrics

    def training_step(self, ep: int) -> dict:
        """Execute single training step"""
        # Sample using last policy
        samples, rewards = self.sample_batch(ep)

        # Compute weights
        weights = torch.exp(self.beta * rewards).to(self.device)

        # Compute and optimize loss
        self.optimizer.zero_grad()
        loss, metrics = self.compute_loss(samples, weights)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.net_model.parameters(),
            self.grad_clip
        )

        self.optimizer.step()
        self.scheduler.step()

        # Update metrics
        metrics.update({
            'reward': rewards.mean().item(),
            'max_reward': rewards.max().item()
        })

        return metrics

    def update_last_policy(self):
        """Update last policy with current model weights"""
        self.last_policy.load_state_dict(self.net_model.state_dict())
        self.last_policy.eval()

    def train(self, num_epochs: int, steps_per_epoch: int):
        """Training loop"""
        for ep in range(num_epochs):
            running_metrics = []

            # Training steps
            pbar = tqdm(range(steps_per_epoch), desc=f'Epoch {ep}')
            for _ in pbar:
                metrics = self.training_step(ep)
                running_metrics.append(metrics)

                # Update progress bar
                pbar.set_postfix({
                    'loss': metrics['total_loss'],
                    'reward': metrics['reward']
                })

            # Update last policy
            self.update_last_policy()

            # Compute epoch metrics
            epoch_metrics = {
                k: sum(d[k] for d in running_metrics) / len(running_metrics)
                for k in running_metrics[0].keys()
            }

            # Log metrics
            if self.use_wandb:
                wandb.log(
                    {f'train/{k}': v for k, v in epoch_metrics.items()},
                    step=ep
                )

    def save_checkpoint(self, path: str):
        """Save training checkpoint"""
        checkpoint = {
            'net_model': self.net_model.state_dict(),
            'last_policy': self.last_policy.state_dict(),
            'optim': self.optimizer.state_dict(),
            'sched': self.scheduler.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, path)


def main():
    config = {
        'learning_rate': 2e-4,
        'warmup_steps': 5000,
        'w2_coefficient': 1.0,
        'temperature': 0.5,
        'grad_clip': 1.0,
        'batch_size': 128,
        'text_prompts': ["An image of dog", "Not an image of dog"],
        'use_wandb': True,
        'wandb_project': 'cifar10-flow-matching',
        'run_name': 'orw-cfm-w2',
        'parallel': False,
        'savedir': './results',
        'ref_path': f'./pretrained/fm_cifar_{400000}.pt'
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = ORWCFMTrainer(model, config, device)
    trainer.load_pretrained(config['ref_path'])

    trainer.train(
        num_epochs=1000,
        steps_per_epoch=int(1e4)
    )


if __name__ == "__main__":
    main()