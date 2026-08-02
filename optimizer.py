# optimizer.py
import torch
import torch.optim as optim
from transformers.optimization import get_cosine_schedule_with_warmup

class OptimizerGroup(optim.Optimizer):
    def __init__(self, *optimizers):
        self.optimizers = optimizers
        # Use the wrapped optimizers' param_group dicts by reference so
        # scheduler LR updates (and load_state_dict) reach AdamW/Muon.
        param_groups = [pg for opt in optimizers for pg in opt.param_groups]
        super().__init__(param_groups, defaults={})
        self.param_groups = param_groups

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for optimizer in self.optimizers:
            optimizer.step()
        return loss

    def zero_grad(self, set_to_none=False):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        # Parent Optimizer.state is unused; real moments live in the children.
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict):
        if "optimizers" not in state_dict:
            raise ValueError(
                "optimizer.pt is missing nested 'optimizers' state (saved with a "
                "broken OptimizerGroup.state_dict). Delete it or re-save a checkpoint "
                "after this fix; resume would otherwise restart with fresh moments."
            )
        if len(state_dict["optimizers"]) != len(self.optimizers):
            raise ValueError(
                f"optimizer checkpoint has {len(state_dict['optimizers'])} nested "
                f"optimizers but this run has {len(self.optimizers)}"
            )
        for opt, sd in zip(self.optimizers, state_dict["optimizers"]):
            opt.load_state_dict(sd)


def build_optimizer(model, learning_rate):
    if isinstance(learning_rate, tuple):
        lr_muon, lr_adam = learning_rate
    else:
        lr_muon = learning_rate
        lr_adam = 1e-3
    adam = optim.AdamW([p for p in model.parameters() if p.ndim != 2], 
        lr=lr_adam, 
        weight_decay=0.01,
        betas=(0.9, 0.95),
        eps=1e-8,
        )
    muon = optim.Muon([p for p in model.parameters() if p.ndim == 2], 
        lr=lr_muon # e.g 2e-2
        )

    return OptimizerGroup(adam, muon)


def build_scheduler(optimizer, total_steps, warmup_steps):
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
