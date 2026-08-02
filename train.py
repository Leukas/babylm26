# train.py
import argparse
import json
from datasets import load_from_disk, load_dataset
import numpy as np
import os
import random
import shutil
from torch.utils.data import DataLoader
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3Config
from transformers.models.deberta_v2.tokenization_deberta_v2 import DebertaV2Tokenizer
from transformers.trainer_utils import set_seed
import torch
from tqdm import tqdm
from trl.data_utils import pack_dataset
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

try:
    import wandb
except ImportError:
    wandb = None

from optimizer import build_optimizer, build_scheduler


def parse_lr(value):
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise argparse.ArgumentTypeError(
        f"learning rate must be one value or two comma-separated values, got {value!r}"
    )


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='leukas/Qwenfood')
parser.add_argument('--vocab', '--tokenizer', type=str, default='leukas/Qwenfant-0.1B')
parser.add_argument('--model_size', type=str, default='small', choices=['small', 'base', 'large'])
parser.add_argument('--bs', '--batch_size', type=int, default=16, dest='batch_size', help='Micro batch size for training')
parser.add_argument('--grad_acc', '--gradient_accumulation_steps', type=int, default=1)
parser.add_argument('--lr', '--learning_rate', type=parse_lr, default=(2e-2, 5e-3), dest='learning_rate',
    help='Muon and Adam LR as "muon,adam" (e.g. "2e-2,5e-3") or a single LR for Muon',
)
parser.add_argument('--max_grad_norm', type=float, default=1.0,
    help='Maximum gradient norm for clipping (0 to disable)',
)
parser.add_argument('--warmup_steps', type=int, default=1000)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--cpus', type=int, default=20)
parser.add_argument('--eval_steps', type=int, default=2000)
parser.add_argument('--save_steps', type=int, default=2000)
parser.add_argument('--log_steps', type=int, default=200)
parser.add_argument('--max_seq_len', type=int, default=2048, help='Maximum sequence length for packing.')
parser.add_argument('--attn_len', type=int, default=2048, help='Qwen 3 sliding window attention length.')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--save_babylm_checkpoints', action='store_true', help='Save checkpoints needed for BabyLM submission.')
parser.add_argument('--output_path', type=str, default='./models/qwenfant-100m')
parser.add_argument('--compile', action='store_true', help='Use torch.compile for training')
parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
parser.add_argument('--resume', nargs='?', const='auto', default=None,
    help='Resume training from a checkpoint directory. If set without a value, uses the latest checkpoint under --output_path.',
)
parser.add_argument('--flop_estimate', action='store_true',
    help='Estimate total training FLOPs from one forward-backward pass times total optimizer steps, then exit',
)
args = parser.parse_args()

def build_model(args):
    if args.model_size == 'small':
        hidden_size = 768
        num_hidden_layers = 12
        num_heads = 12
    elif args.model_size == 'base':
        hidden_size = 1024
        num_hidden_layers = 24
        num_heads = 16
    elif args.model_size == 'large':
        hidden_size = 1536
        num_hidden_layers = 24
        num_heads = 16

    config = Qwen3Config(
        vocab_size=args.vocab_size,
        pad_token_id=args.pad_token_id,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        head_dim=hidden_size // num_heads,
        intermediate_size=hidden_size * 4,
        max_position_embeddings=args.max_seq_len,
        rms_norm_eps=1e-6,
        sliding_window=args.attn_len,
        attn_implementation='flash_attention_3',
    )
    model = Qwen3ForCausalLM(config)
    return model

def eval_loop(model, tokenizer, dataset, args):
    collator = DataCollatorForLanguageModeling(
        pad_token_id=tokenizer.pad_token_id,
        completion_only_loss=False,
        padding_free=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collator,
        shuffle=False,
        num_workers=args.cpus,
        pin_memory=True,
    )

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    with torch.inference_mode():
        for batch in dataloader:
            batch = {k: v.to(device=model.device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                outputs = model(**batch)

            labels = batch["labels"]
            shift_logits = outputs.logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            mask = shift_labels != -100
            num_tokens = mask.sum().item()

            total_loss += outputs.loss.item() * num_tokens
            total_tokens += num_tokens
            predictions = shift_logits.argmax(dim=-1)
            total_correct += ((predictions == shift_labels) & mask).sum().item()

    model.train()

    return {
        "loss": total_loss / max(total_tokens, 1),
        "accuracy": total_correct / max(total_tokens, 1),
    }


def _format_flops(flops):
    for suffix in ("", "K", "M", "G", "T", "P", "E"):
        if abs(flops) < 1000.0:
            return f"{flops:.3f}{suffix}FLOPs" if suffix else f"{flops:.0f}FLOPs"
        flops /= 1000.0
    return f"{flops:.3f}ZFLOPs"


def _measure_fwd_bwd_flops(model, batch, args):
    model.zero_grad(set_to_none=True)
    batch = {k: v.to(device=model.device, non_blocking=True) for k, v in batch.items()}

    def _run_fwd_bwd():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            outputs = model(**batch)
            loss = outputs.loss
        (loss / args.grad_acc).backward()

    try:
        from torch.utils.flop_counter import FlopCounterMode

        with FlopCounterMode(display=False) as flop_counter:
            _run_fwd_bwd()
        return flop_counter.get_total_flops()
    except ImportError:
        from torch.profiler import profile, ProfilerActivity

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        with profile(activities=activities, with_flops=True) as prof:
            _run_fwd_bwd()
        return sum(event.flops for event in prof.key_averages() if event.flops > 0)


def estimate_training_flops(model, tokenizer, dataset, args):
    collator = DataCollatorForLanguageModeling(
        pad_token_id=tokenizer.pad_token_id,
        completion_only_loss=False,
        padding_free=True,
    )
    dataloader = DataLoader(
        dataset["train"],
        batch_size=args.batch_size,
        collate_fn=collator,
        shuffle=True,
        num_workers=args.cpus,
        pin_memory=True,
    )

    micro_steps_per_epoch = len(dataloader)
    steps_per_epoch = (micro_steps_per_epoch + args.grad_acc - 1) // args.grad_acc
    total_steps = args.epochs * steps_per_epoch

    model.train()
    model = model.to(device="cuda:0")
    batch = next(iter(dataloader))
    flops_per_pass = _measure_fwd_bwd_flops(model, batch, args)
    total_flops = flops_per_pass * total_steps

    seq_len = batch["input_ids"].shape[1]
    print("FLOP estimate (one forward-backward pass x total optimizer steps):")
    print(f"  batch_size={args.batch_size}, seq_len={seq_len}, grad_acc={args.grad_acc}")
    print(f"  FLOPs per forward-backward pass: {_format_flops(flops_per_pass)}")
    print(f"  optimizer steps per epoch: {steps_per_epoch}")
    print(f"  total optimizer steps: {total_steps}")
    print(f"  estimated total training FLOPs: {_format_flops(total_flops)} ({total_flops:.3e})")
    if args.grad_acc > 1:
        micro_steps_total = micro_steps_per_epoch * args.epochs
        micro_total_flops = flops_per_pass * micro_steps_total
        print(
            f"  note: with grad_acc={args.grad_acc}, total micro-batch passes are "
            f"{micro_steps_total}; micro-batch FLOP estimate: {_format_flops(micro_total_flops)}"
        )
    return total_flops


def build_babylm_checkpoint_schedule(total_steps):
    # 1M, 2M, ..., 10M, 20M, ..., 100M, 200M, ..., 1000M word milestones
    steps_1m = np.linspace(total_steps // 1000, total_steps // 100, 10).astype(int)
    steps_10m = np.linspace(total_steps // 100, total_steps // 10, 10).astype(int)
    steps_100m = np.linspace(total_steps // 10, total_steps, 10).astype(int)
    checkpoint_steps = list(steps_1m) + list(steps_10m)[1:] + list(steps_100m)[1:]

    names = [f"chck_{i}M" for i in range(1, 11)]
    names += [f"chck_{i * 10}M" for i in range(2, 11)]
    names += [f"chck_{i * 100}M" for i in range(2, 11)]
    return dict(zip(checkpoint_steps, names))


def _fix_tokenizer_config(save_dir):
    """Remove extra_special_tokens list; breaks loading on transformers 4.x."""
    config_path = os.path.join(save_dir, "tokenizer_config.json")
    if not os.path.isfile(config_path):
        return
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    if isinstance(config.get("extra_special_tokens"), list):
        del config["extra_special_tokens"]
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
            f.write("\n")


def _fix_tokenizer_byte_fallback(save_dir: str):
    """Ensure model.byte_fallback=true in tokenizer.json (HF conversion can default to false)."""
    tok_path = os.path.join(save_dir, "tokenizer.json")
    if not os.path.isfile(tok_path):
        return
    try:
        with open(tok_path, encoding="utf-8") as f:
            data = json.load(f)
        model = data.get("model")
        if not isinstance(model, dict):
            return
        if model.get("byte_fallback") is True:
            return
        model["byte_fallback"] = True
        with open(tok_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
    except Exception:
        # Best-effort: don't fail checkpointing because of tokenizer.json quirks.
        return

def _checkpoint_has_model_files(checkpoint_dir: str) -> bool:
    if not os.path.isdir(checkpoint_dir):
        return False
    # HF model save_pretrained may produce different filenames depending on format.
    for fname in ("config.json", "model.safetensors", "pytorch_model.bin", "pytorch_model.bin.index.json"):
        if os.path.exists(os.path.join(checkpoint_dir, fname)):
            return True
    return False


def _find_latest_checkpoint(output_path: str) -> str | None:
    if not output_path or not os.path.isdir(output_path):
        return None
    candidates = []
    for name in os.listdir(output_path):
        ckpt_dir = os.path.join(output_path, name)
        if not os.path.isdir(ckpt_dir):
            continue
        state_path = os.path.join(ckpt_dir, "train_state.json")
        if os.path.isfile(state_path):
            try:
                with open(state_path, encoding="utf-8") as f:
                    state = json.load(f)
                step = int(state.get("global_step", -1))
            except Exception:
                step = -1
            if step >= 0 and _checkpoint_has_model_files(ckpt_dir):
                candidates.append((step, ckpt_dir))
                continue
        # Back-compat: allow resuming from a plain save_pretrained directory
        if _checkpoint_has_model_files(ckpt_dir):
            # Try to parse "checkpoint-123" if present; otherwise treat as step=-1.
            step = -1
            if name.startswith("checkpoint-"):
                try:
                    step = int(name.split("-", 1)[1])
                except Exception:
                    step = -1
            candidates.append((step, ckpt_dir))
    if not candidates:
        return None
    # Prefer highest known global_step, then newest mtime.
    candidates.sort(key=lambda x: (x[0], os.path.getmtime(x[1])))
    return candidates[-1][1]


def _load_checkpoint_state(resume: str | None, output_path: str):
    if not resume:
        return None, None
    if resume == "auto":
        ckpt_dir = _find_latest_checkpoint(output_path)
        if ckpt_dir is None:
            raise FileNotFoundError(
                f"--resume was set but no checkpoint was found under output_path={output_path!r}"
            )
    else:
        ckpt_dir = resume
        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(f"checkpoint directory not found: {ckpt_dir!r}")
    state_path = os.path.join(ckpt_dir, "train_state.json")
    state = None
    if os.path.isfile(state_path):
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
    return ckpt_dir, state


def _save_train_state(save_dir: str, train_globals: dict):
    state = {
        "global_step": int(train_globals.get("global_step", 0)),
        "epoch": int(train_globals.get("epoch", 0)),
        "micro_step": int(train_globals.get("micro_step", 0)),
        "python_random_state": repr(random.getstate()),
        "numpy_random_state": repr(np.random.get_state()),
        "torch_rng_state": torch.get_rng_state().tolist(),
    }
    if torch.cuda.is_available():
        try:
            state["torch_cuda_rng_state_all"] = [s.tolist() for s in torch.cuda.get_rng_state_all()]
        except Exception:
            pass
    with open(os.path.join(save_dir, "train_state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
        f.write("\n")


def save_checkpoint(
    model,
    tokenizer,
    output_path,
    global_step,
    train_globals,
    *,
    optimizer=None,
    scheduler=None,
    checkpoint_name=None,
    keep_previous=False,
):
    name = checkpoint_name or f"checkpoint-{global_step}"
    save_dir = os.path.join(output_path, name)
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    _fix_tokenizer_config(save_dir)
    _fix_tokenizer_byte_fallback(save_dir)
    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
    if scheduler is not None:
        torch.save(scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt"))
    _save_train_state(save_dir, train_globals)
    print(f"Saved checkpoint to {save_dir}")

    if keep_previous:
        train_globals.setdefault("saved_checkpoints", set()).add(save_dir)
        return

    prev_dir = train_globals.get("last_checkpoint_dir")
    if prev_dir and prev_dir != save_dir and os.path.isdir(prev_dir):
        shutil.rmtree(prev_dir)
        print(f"Removed previous checkpoint at {prev_dir}")
    train_globals["last_checkpoint_dir"] = save_dir


def train_loop(
    model,
    tokenizer,
    batch,
    optimizer,
    scheduler,
    train_globals,
    dataset,
    args,
):
    batch = {k: v.to(device=model.device, non_blocking=True) for k, v in batch.items()}
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        outputs = model(**batch)
        loss = outputs.loss
    (loss / args.grad_acc).backward()

    train_globals["loss_accum"] += loss.item()
    train_globals["loss_accum_count"] += 1

    micro_step = train_globals["micro_step"]
    should_step = (micro_step % args.grad_acc == 0) or (
        micro_step == train_globals["micro_steps_per_epoch"]
    )
    if not should_step:
        return

    if args.max_grad_norm > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=args.max_grad_norm
        )
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    train_globals["global_step"] += 1
    global_step = train_globals["global_step"]

    train_globals["pbar"].set_postfix(
        loss=f"{loss.item():.3f}",
        epoch=f"{train_globals['epoch'] + 1}/{args.epochs}",
        lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        grad_norm=f"{grad_norm:.2e}",
    )
    train_globals["pbar"].update(1)

    if global_step % args.log_steps == 0:
        avg = train_globals["loss_accum"] / max(train_globals["loss_accum_count"], 1)
        print(f"Step {global_step}/{train_globals['total_steps']} loss: {avg:.4f}")
        if train_globals["wandb_enabled"]:
            wandb.log(
                {
                    "global_step": global_step,
                    "train/loss": avg,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/grad_norm": float(grad_norm),
                    "train/epoch": train_globals["epoch"] + 1,
                    "train/step_fraction": global_step / max(train_globals["total_steps"], 1),
                },
            )
        train_globals["loss_accum"] = 0.0
        train_globals["loss_accum_count"] = 0

    if global_step % args.eval_steps == 0:
        eval_metrics = eval_loop(model, tokenizer, dataset["test"], args)
        eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
        print(f"Step {global_step}/{train_globals['total_steps']} eval: {eval_metrics}")
        if train_globals["wandb_enabled"]:
            wandb.log({"global_step": global_step, **eval_metrics})

    if args.output_path:
        if args.save_babylm_checkpoints:
            babylm_steps = train_globals.get("babylm_checkpoint_steps", {})
            if global_step in babylm_steps:
                save_checkpoint(
                    model,
                    tokenizer,
                    args.output_path,
                    global_step,
                    train_globals,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    checkpoint_name=babylm_steps[global_step],
                    keep_previous=True,
                )
        elif global_step % args.save_steps == 0:
            save_checkpoint(
                model,
                tokenizer,
                args.output_path,
                global_step,
                train_globals,
                optimizer=optimizer,
                scheduler=scheduler,
            )



def train_model(model, tokenizer, dataset, args):
    if args.flop_estimate:
        estimate_training_flops(model, tokenizer, dataset, args)
        return

    ckpt_dir, ckpt_state = _load_checkpoint_state(args.resume, args.output_path)
    if ckpt_dir is not None:
        print(f"Resuming from checkpoint: {ckpt_dir}")
        model = Qwen3ForCausalLM.from_pretrained(ckpt_dir, attn_implementation="flash_attention_3")

    train_dataset = dataset["train"]
    collator = DataCollatorForLanguageModeling(
        pad_token_id=tokenizer.pad_token_id,
        completion_only_loss=False,
        padding_free=True,
    )
    def make_dataloader(epoch: int):
        # Deterministic shuffle per epoch for resumability.
        g = torch.Generator()
        g.manual_seed(args.seed + epoch)
        return DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            collate_fn=collator,
            shuffle=True,
            generator=g,
            num_workers=args.cpus,
            pin_memory=True,
        )
    dataloader = make_dataloader(epoch=0)

    model.train()
    model = model.to(device="cuda:0")

    if args.compile:
        model = torch.compile(model)

    micro_steps_per_epoch = len(dataloader)
    steps_per_epoch = (micro_steps_per_epoch + args.grad_acc - 1) // args.grad_acc
    total_steps = args.epochs * steps_per_epoch

    optimizer = build_optimizer(model, args.learning_rate)
    # Schedule over optimizer steps (not micro-batches).
    scheduler = build_scheduler(optimizer, total_steps, args.warmup_steps)

    def _move_optimizer_state_to_device(opt, device):
        # Nested OptimizerGroup: move each child optimizer's state tensors.
        children = getattr(opt, "optimizers", None)
        if children is None:
            children = (opt,)
        for child in children:
            for state in child.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device=device, non_blocking=True)

    babylm_checkpoint_steps = {}
    if args.save_babylm_checkpoints:
        babylm_checkpoint_steps = build_babylm_checkpoint_schedule(total_steps)
        print(
            "BabyLM checkpoints: "
            + ", ".join(
                f"{name}@{step}"
                for step, name in sorted(babylm_checkpoint_steps.items())
            )
        )

    train_globals = {
        "loss_accum": 0.0,
        "loss_accum_count": 0,
        "micro_steps_per_epoch": micro_steps_per_epoch,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "epoch": 0,
        "micro_step": 0,
        "global_step": 0,
        "pbar": tqdm(total=total_steps, desc="Training", unit="step", leave=True),
        "wandb_enabled": args.wandb and wandb is not None,
        "last_checkpoint_dir": None,
        "babylm_checkpoint_steps": babylm_checkpoint_steps,
        "saved_checkpoints": set(),
    }

    start_epoch = 0
    start_micro_step = 0
    if ckpt_dir is not None:
        opt_path = os.path.join(ckpt_dir, "optimizer.pt")
        sch_path = os.path.join(ckpt_dir, "scheduler.pt")
        if os.path.isfile(opt_path):
            try:
                optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
                _move_optimizer_state_to_device(optimizer, device=model.device)
            except ValueError as e:
                print(
                    f"WARNING: could not restore optimizer state from {opt_path}: {e} "
                    "Continuing with fresh optimizer moments (expect a temporary loss spike)."
                )
        else:
            print(
                f"WARNING: no optimizer.pt in {ckpt_dir}; "
                "continuing with fresh optimizer moments (expect a temporary loss spike)."
            )
        if os.path.isfile(sch_path):
            scheduler.load_state_dict(torch.load(sch_path, map_location="cpu"))
        else:
            print(
                f"WARNING: no scheduler.pt in {ckpt_dir}; "
                "LR schedule restarts from step 0 (warmup)."
            )
        if isinstance(ckpt_state, dict):
            train_globals["global_step"] = int(ckpt_state.get("global_step", 0))
            start_epoch = int(ckpt_state.get("epoch", 0))
            start_micro_step = int(ckpt_state.get("micro_step", 0))
            # Best-effort RNG restoration (kept intentionally non-fatal).
            try:
                if "torch_rng_state" in ckpt_state:
                    torch.set_rng_state(torch.tensor(ckpt_state["torch_rng_state"], dtype=torch.uint8))
                if torch.cuda.is_available() and "torch_cuda_rng_state_all" in ckpt_state:
                    states = [
                        torch.tensor(s, dtype=torch.uint8) for s in ckpt_state["torch_cuda_rng_state_all"]
                    ]
                    torch.cuda.set_rng_state_all(states)
            except Exception:
                pass

        # Rebuild progress bar to reflect already-completed steps.
        train_globals["pbar"].close()
        train_globals["pbar"] = tqdm(
            total=total_steps,
            desc="Training",
            unit="step",
            leave=True,
            initial=train_globals["global_step"],
        )

    for epoch in range(start_epoch, args.epochs):
        train_globals["epoch"] = epoch
        dataloader = make_dataloader(epoch=epoch)
        for micro_step, batch in enumerate(dataloader, start=1):
            if epoch == start_epoch and micro_step <= start_micro_step:
                continue
            train_globals["micro_step"] = micro_step
            train_loop(
                model, tokenizer, batch, optimizer, scheduler,
                train_globals, dataset, args,
            )

    train_globals["pbar"].close()

    final_step = train_globals["global_step"]
    if args.output_path:
        if args.save_babylm_checkpoints:
            babylm_name = babylm_checkpoint_steps.get(final_step)
            final_dir = (
                os.path.join(args.output_path, babylm_name) if babylm_name else None
            )
            if babylm_name and final_dir not in train_globals["saved_checkpoints"]:
                save_checkpoint(
                    model,
                    tokenizer,
                    args.output_path,
                    final_step,
                    train_globals,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    checkpoint_name=babylm_name,
                    keep_previous=True,
                )
        elif total_steps % args.save_steps != 0:
            save_checkpoint(
                model,
                tokenizer,
                args.output_path,
                final_step,
                train_globals,
                optimizer=optimizer,
                scheduler=scheduler,
            )


def main():
    args = parser.parse_args()
    set_seed(args.seed, deterministic=True)
    if args.wandb:
        if wandb is None:
            raise ImportError("wandb is not installed but --wandb was set. Install wandb or run without --wandb.")
        output_dir = os.path.basename(os.path.normpath(args.output_path))
        wandb.init(
            project='bb26',
            name=output_dir if args.output_path else None,
            config=vars(args),
        )
        # One x-axis for all metrics; avoid logging train/step (conflicts with step=).
        wandb.define_metric("global_step", hidden=True)
        wandb.define_metric("train/*", step_metric="global_step")
        wandb.define_metric("eval/*", step_metric="global_step")

    tokenizer = DebertaV2Tokenizer.from_pretrained(
        args.vocab,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        cls_token="<s>",
        sep_token="</s>",
        mask_token="<unk>",
    )

    args.pad_token_id = tokenizer.pad_token_id
    args.vocab_size = len(tokenizer)
    model = build_model(args)
    try:
        dataset = load_dataset(args.dataset)
    except FileNotFoundError:
        dataset = load_from_disk(args.dataset)
    for split in dataset.keys():
        column_names = dataset[split].column_names
        dataset[split] = dataset[split].map(
            lambda x: {"input_ids": tokenizer(x["text"], add_special_tokens=False)["input_ids"]},
            batched=True,
            num_proc=args.cpus,
            remove_columns=column_names,
        )
        dataset[split] = pack_dataset(dataset[split], args.max_seq_len, strategy="bfd_split", map_kwargs={'num_proc': args.cpus})
    train_model(model, tokenizer, dataset, args)
    if args.wandb and wandb is not None:
        wandb.finish()

if __name__ == '__main__':
    main()