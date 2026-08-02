# BabyLM26

This repo contains our code for reproducing our BabyLM2026 submission (link to paper TBA).
You can find our trained models here: https://huggingface.co/collections/leukas/babylm-2026

## Prerequisites
The scripts have been tested on:
- Python 3.12
- PyTorch 2.10
- Transformers 5.5.4
- TRL 1.2.0
- [flash-attn 3.0.0](https://github.com/Dao-AILab/flash-attention)

The training should run without flash attention, but it will not be identical since examples are packed a 2nd time for training, and flash attention 3 will prevent attention between these examples, and the default SDPA will not. Changing the `--max_seq_len` flag can be adjusted to account for this, or by removing packing.

## Training
To train Qwenfant-0.1B model, for example, run:
```bash
python train.py --tokenizer leukas/Qwenfant-0.1B --output_path ./models/qwenfant-100m/ --dataset leukas/Qwenfood --save_babylm_checkpoints --model_size small
```


All flags are explained here:
```
--dataset                   Training dataset path or Hugging Face id. Default: leukas/Qwenfood
--vocab, --tokenizer        Tokenizer path or name. Default: leukas/Qwenfant-0.1B
--model_size                Model size: small, base, or large. Default: small
--bs, --batch_size          Micro batch size for training. Default: 16
--grad_acc, --gradient_accumulation_steps
                            Gradient accumulation steps. Default: 1
--lr, --learning_rate       Muon and Adam LR as "muon,adam" (e.g. "2e-2,5e-3") or a single LR for Muon. Default: 2e-2,5e-3
--max_grad_norm             Maximum gradient norm for clipping (0 to disable). Default: 1.0
--warmup_steps              Warmup steps for the learning rate schedule. Default: 1000
--epochs                    Number of training epochs. Default: 10
--cpus                      Number of CPU workers for data loading. Default: 20
--eval_steps                Run evaluation every N steps. Default: 2000
--save_steps                Save a checkpoint every N steps. Default: 2000
--log_steps                 Log training metrics every N steps. Default: 200
--max_seq_len               Maximum sequence length for packing. Default: 2048
--attn_len                  Qwen 3 sliding window attention length. Default: 2048
--seed                      Random seed. Default: 0
--save_babylm_checkpoints   Save checkpoints needed for BabyLM submission.
--output_path               Output directory for model checkpoints and logs. Default: ./models/qwenfant-100m
--compile                   Use torch.compile for training.
--wandb                     Report metrics to Weights & Biases.
--resume                    Resume training from a checkpoint directory. If set without a value, uses the latest checkpoint under --output_path.
--flop_estimate             Estimate total training FLOPs from one forward-backward pass times total optimizer steps, then exit.
```
