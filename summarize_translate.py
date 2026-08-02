# summarize_translate.py — LLM summarize or translate with Qwen2.5-Instruct
import argparse
import json
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import count_bytes, eng_33M_bytes, filter_to_byte_budget

MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit/"

LANG_DATASETS = {
    "english": "BabyLM-community/BabyLM-2026-Strict",
    "dutch": "BabyLM-community/babylm-nld",
    "chinese": "BabyLM-community/babylm-zho",
}

def _text_byte_len(text):
    return len(text.encode("utf-8"))


def _split_text_by_bytes(text, chunk_size):
    encoded = text.encode("utf-8")
    if len(encoded) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(encoded):
        end = min(start + chunk_size, len(encoded))
        if end < len(encoded):
            while end > start and (encoded[end] & 0xC0) == 0x80:
                end -= 1
            if end == start:
                end = start + 1
                while end < len(encoded) and (encoded[end] & 0xC0) == 0x80:
                    end += 1
        chunks.append(encoded[start:end].decode("utf-8"))
        start = end
    return chunks


def _pack_texts_by_words(texts, chunk_size):
    chunks = []
    current = []
    current_len = 0

    for text in texts:
        words = text.split()
        n_words = len(words)

        if n_words > chunk_size:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            for start in range(0, n_words, chunk_size):
                chunks.append(" ".join(words[start : start + chunk_size]))
            continue

        if current_len > 0 and current_len + n_words > chunk_size:
            chunks.append(" ".join(current))
            current = words
            current_len = n_words
        else:
            current.extend(words)
            current_len += n_words

    if current:
        chunks.append(" ".join(current))
    return chunks


def _pack_texts_by_bytes(texts, chunk_size):
    chunks = []
    current_parts = []
    current_len = 0

    def flush():
        nonlocal current_parts, current_len
        if current_parts:
            chunks.append(" ".join(current_parts))
            current_parts = []
            current_len = 0

    for text in texts:
        text_bytes = _text_byte_len(text)

        if text_bytes > chunk_size:
            flush()
            chunks.extend(_split_text_by_bytes(text, chunk_size))
            continue

        sep_bytes = 1 if current_parts else 0
        if current_len > 0 and current_len + sep_bytes + text_bytes > chunk_size:
            flush()
            current_parts = [text]
            current_len = text_bytes
        else:
            current_parts.append(text)
            current_len += sep_bytes + text_bytes

    flush()
    return chunks


def pack_texts(texts, chunk_size, mode="words"):
    if mode == "bytes":
        return _pack_texts_by_bytes(texts, chunk_size)
    return _pack_texts_by_words(texts, chunk_size)


def build_summarize_prompt(text, summary_length):
    return (
        f"Shorten the following text to {summary_length} words or less. "
        f"Don't output any other text:\n\n{text}"
    )


def build_translate_prompt(text, tgt_lang):
    return f"Translate the following text to {tgt_lang}. Don't output any other text:\n\n{text}"


def extract_response(decoded):
    decoded = decoded.split("\nassistant\n")[1].strip()
    if decoded.startswith("Sure,"):
        try:
            return decoded.split(":")[1].strip()
        except IndexError:
            print("Cannot extract response from: ", decoded)
            return decoded
    return decoded


def generate_batch(model, tokenizer, prompts, device, max_new_tokens, num_beams):
    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        padding=True,
        add_generation_prompt=True,
        return_dict=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                pad_token_id=tokenizer.pad_token_id,
            )
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return [extract_response(text) for text in decoded]


def load_summarize_texts(lang):
    dataset = load_dataset(LANG_DATASETS[lang])
    return dataset["train"]["text"]


def load_translate_texts(dataset_name, no_translate):
    if dataset_name == "summaries":
        dataset = load_dataset("json", data_files="summaries.jsonl")
        return dataset["train"]["summary"]

    dataset = load_dataset("BabyLM-community/BabyLM-2026-Strict")["train"]
    dataset = dataset.map(count_bytes, batched=True)
    print(dataset)
    if not no_translate:
        dataset = filter_to_byte_budget(dataset, eng_33M_bytes)
    return dataset["text"]


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize or translate text with Qwen2.5-Instruct.")
    parser.add_argument("--mode", type=str, default="summarize", choices=["summarize", "translate"], help="summarize: shorten texts; translate: LLM-translate texts")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--chunk_size", type=int, default=512, help="Length of chunk of examples for packing.")
    parser.add_argument("--pack_mode", type=str, default="words", choices=["words", "bytes"], help="pack chunks by word count or UTF-8 byte count")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--summary_length", type=int, default=200)
    parser.add_argument("--lang", type=str, default="english", choices=["english", "dutch", "chinese"], help="source language dataset for --mode summarize")
    parser.add_argument("--dataset", type=str, default="summaries", choices=["summaries", "original"], help="input for --mode translate (summaries.jsonl or BabyLM English)")
    parser.add_argument("--tgt_lang", type=str, default="Dutch")
    parser.add_argument("--pack", action="store_true", help="pack texts into chunks before translate (always on for summarize)")
    parser.add_argument("--no_translate", action="store_true", help="with --mode translate: write packed/raw texts and exit")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "summarize":
        texts = load_summarize_texts(args.lang)
        texts_to_process = pack_texts(texts, args.chunk_size, mode=args.pack_mode)
        print(f"Packed {len(texts)} texts into {len(texts_to_process)} chunks ({args.pack_mode})")
        output_path = args.output_path or "summaries.jsonl"
        out_key = "summary"
        prompt_fn = lambda text: build_summarize_prompt(text, args.summary_length)
    else:
        texts = load_translate_texts(args.dataset, args.no_translate)
        if args.pack:
            texts_to_process = pack_texts(texts, args.chunk_size, mode=args.pack_mode)
            print(f"Packed {len(texts)} texts into {len(texts_to_process)} chunks ({args.pack_mode})")
        else:
            texts_to_process = texts

        if args.no_translate:
            output_path = args.output_path or "texts.jsonl"
            with open(output_path, "w") as f:
                for text in texts_to_process:
                    f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            print(f"Saved {len(texts_to_process)} texts to {output_path}")
            return

        base = args.output_path or "translated_summaries.jsonl"
        output_path = f"{args.tgt_lang.lower()}_{base}"
        out_key = args.tgt_lang.lower()
        prompt_fn = lambda text: build_translate_prompt(text, args.tgt_lang)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        attn_implementation="flash_attention_3",
    )
    device = next(model.parameters()).device
    model.eval()

    n_written = 0
    with open(output_path, "w") as f:
        for start in tqdm(range(0, len(texts_to_process), args.batch_size)):
            batch = texts_to_process[start : start + args.batch_size]
            outputs = generate_batch(
                model,
                tokenizer,
                [prompt_fn(text) for text in batch],
                device,
                args.max_new_tokens,
                args.num_beams,
            )
            for src, out in zip(batch, outputs):
                f.write(json.dumps({"text": src, out_key: out}, ensure_ascii=False) + "\n")
            f.flush()
            n_written += len(outputs)

    print(f"Saved {n_written} outputs to {output_path}")


if __name__ == "__main__":
    main()
