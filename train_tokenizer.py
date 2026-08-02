# train_tokenizer.py
from sentencepiece import SentencePieceTrainer
from datasets import load_from_disk
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='data/baseline_dataset')
parser.add_argument('--model_prefix', type=str, default='tokenizer')
parser.add_argument('--vocab_size', type=int, default=40000)
args = parser.parse_args()

dataset = load_from_disk(args.dataset)

SentencePieceTrainer.train(
    sentence_iterator=iter(dataset['train']['text']),
    model_prefix=args.model_prefix,
    vocab_size=args.vocab_size,
    character_coverage=0.9995,
    # character_coverage=1.0,
    max_sentencepiece_length=9,
    byte_fallback=True,
    model_type='bpe',
    pad_piece='<pad>',
    bos_piece='<s>',
    eos_piece='</s>',
    unk_piece='<unk>',
    pad_id=0,
    bos_id=1,
    eos_id=2,
    unk_id=3,
)
