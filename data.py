# data.py
import argparse
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from summarize_translate import pack_texts

FLORES_DATASET = "openlanguagedata/flores_plus"
FLORES_SPLIT = "dev"
FLORES_LANG_CODES = {
    "eng": "eng_Latn",
    "nld": "nld_Latn",
    "zho": "cmn_Hans",
}

byte_premiums = {
    'eng': 1,
    'nld': 1.051606,
    'zho': 0.935966,
}

eng_100M_bytes = 539181484
eng_33M_bytes = eng_100M_bytes // 3
nld_33M_bytes = eng_33M_bytes * byte_premiums['nld']
zho_33M_bytes = eng_33M_bytes * byte_premiums['zho']


def count_bytes(batch):
    return {'bytes': [len(text.encode('utf-8')) for text in batch['text']]}


def total_column_bytes(dataset, column):
    return sum(len(text.encode('utf-8')) for text in dataset[column])


def check_byte_ratio(label, tgt_bytes, src_bytes, premium):
    ratio = tgt_bytes / src_bytes
    exceeds = ratio > premium
    status = 'EXCEEDS premium' if exceeds else 'within premium'
    print(f"{label}: {tgt_bytes:,} / {src_bytes:,} = {ratio:.6f} (premium {premium:.6f}) — {status}")
    return ratio, exceeds


def load_flores_language(lang_key, split=FLORES_SPLIT):
    """One sentence per row for a single language."""
    ds = load_dataset(FLORES_DATASET, FLORES_LANG_CODES[lang_key], split=split)
    return ds.remove_columns([c for c in ds.column_names if c != "text"])


def build_flores_test(split=FLORES_SPLIT):
    """FLORES+ dev/devtest: one monolingual sentence per example."""
    parts = []
    for lang_key in ("eng", "nld", "zho"):
        part = load_flores_language(lang_key, split=split)
        parts.append(part.map(count_bytes, batched=True))
    return concatenate_datasets(parts)


def filter_to_byte_budget(dataset, max_bytes):
    total = 0
    end = 0
    for nbytes in dataset['bytes']:
        if total + nbytes > max_bytes:
            break
        total += nbytes
        end += 1
    return dataset.select(range(end))


def build_baseline_dataset():
    zho_data = load_dataset('BabyLM-community/babylm-zho', split='train')
    nld_data = load_dataset('BabyLM-community/babylm-nld', split='train')
    eng_data = load_dataset('BabyLM-community/babylm-eng', split='train')

    zho_data = zho_data.map(count_bytes, batched=True, num_proc=20)
    nld_data = nld_data.map(count_bytes, batched=True, num_proc=20)
    eng_data = eng_data.map(count_bytes, batched=True, num_proc=20)

    zho_data = filter_to_byte_budget(zho_data, zho_33M_bytes)
    nld_data = filter_to_byte_budget(nld_data, nld_33M_bytes)
    eng_data = filter_to_byte_budget(eng_data, eng_33M_bytes)

    test = build_flores_test()
    all_data = DatasetDict({
        'train': concatenate_datasets([zho_data, nld_data, eng_data]),
        'test': test,
    })

    print(f"Train bytes: {sum(all_data['train']['bytes'])}")
    print(f"Test bytes: {sum(all_data['test']['bytes'])}")

    all_data.save_to_disk('data/baseline_dataset')

    return all_data

def concat_sentences(batch, lang1, lang2, lang3, src_col='src', tgt_col='tgt', tgt2_col='tgt2'):
    new_batch = {'text': []}
    for i in range(len(batch[src_col])):
        src = batch[src_col][i]
        tgt = batch[tgt_col][i]
        tgt2 = batch[tgt2_col][i]
        if i % 6 == 0:
            new_batch['text'].append(f"{src} <{lang2}> {tgt} <{lang3}> {tgt2}")
        elif i % 6 == 1:
            new_batch['text'].append(f"{src} <{lang3}> {tgt2} <{lang2}> {tgt}")
        elif i % 6 == 2:
            new_batch['text'].append(f"{tgt} <{lang1}> {src} <{lang2}> {tgt2}")
        elif i % 6 == 3:
            new_batch['text'].append(f"{tgt} <{lang1}> {tgt2} <{lang3}> {src}")
        elif i % 6 == 4:
            new_batch['text'].append(f"{tgt2} <{lang2}> {tgt} <{lang1}> {src}")
        else: # i % 6 == 5
            new_batch['text'].append(f"{tgt2} <{lang1}> {src} <{lang3}> {tgt}")
    return new_batch


def concat_en_nl_en_zh(en_nl, en_zh, old_col='tgt', new_col='tgt2'):
    if len(en_nl) != len(en_zh):
        raise ValueError(f"Length mismatch: en_nl={len(en_nl)}, en_zh={len(en_zh)}")
    return en_nl.add_column(new_col, en_zh[old_col])


def build_packed_translated_dataset():
    en_nl = load_dataset('json', data_files='data/packed_nl.jsonl', split='train')
    en_zh = load_dataset('json', data_files='data/packed_zh.jsonl', split='train')

    en_bytes = total_column_bytes(en_nl, 'text')
    nl_bytes = total_column_bytes(en_nl, 'dutch')
    zh_bytes = total_column_bytes(en_zh, 'chinese')
    en_bytes_zh = total_column_bytes(en_zh, 'text')
    if en_bytes != en_bytes_zh:
        print(f"Warning: en src bytes differ between en_nl ({en_bytes:,}) and en_zh ({en_bytes_zh:,})")

    print("Byte counts (translations vs English source):")
    check_byte_ratio('nl/en', nl_bytes, en_bytes, byte_premiums['nld'])
    check_byte_ratio('zh/en', zh_bytes, en_bytes, byte_premiums['zho'])

    data = concat_en_nl_en_zh(en_nl, en_zh, old_col='chinese', new_col='chinese')

    train = data.map(
        concat_sentences,
        batched=True,
        fn_kwargs={'lang1': 'english', 'lang2': 'dutch', 'lang3': 'chinese', 'src_col': 'text', 'tgt_col': 'dutch', 'tgt2_col': 'chinese'},
        remove_columns=data.column_names,
        num_proc=20,
    )
    train = train.map(count_bytes, batched=True, num_proc=20)
    test = build_flores_test()

    all_data = DatasetDict({'train': train, 'test': test})
    print(f"Train: {len(all_data['train'])} examples, {sum(all_data['train']['bytes']):,} bytes")
    print(f"Test ({FLORES_DATASET} {FLORES_SPLIT}): {len(all_data['test'])} examples, {sum(all_data['test']['bytes']):,} bytes")
    all_data.save_to_disk('data/packed_translated_dataset')
    return all_data



def build_translated_dataset():
    en_nl = load_dataset('json', data_files='data/en_nl.jsonl', split='train')
    en_zh = load_dataset('json', data_files='data/en_zh.jsonl', split='train')

    en_bytes = total_column_bytes(en_nl, 'src')
    nl_bytes = total_column_bytes(en_nl, 'tgt')
    zh_bytes = total_column_bytes(en_zh, 'tgt')
    en_bytes_zh = total_column_bytes(en_zh, 'src')
    if en_bytes != en_bytes_zh:
        print(f"Warning: en src bytes differ between en_nl ({en_bytes:,}) and en_zh ({en_bytes_zh:,})")

    print("Byte counts (translations vs English source):")
    check_byte_ratio('nl/en', nl_bytes, en_bytes, byte_premiums['nld'])
    check_byte_ratio('zh/en', zh_bytes, en_bytes, byte_premiums['zho'])

    data = concat_en_nl_en_zh(en_nl, en_zh)

    train = data.map(
        concat_sentences,
        batched=True,
        fn_kwargs={'lang1': 'english', 'lang2': 'dutch', 'lang3': 'chinese'},
        remove_columns=data.column_names,
        num_proc=20,
    )
    train = train.map(count_bytes, batched=True, num_proc=20)
    test = build_flores_test()

    all_data = DatasetDict({'train': train, 'test': test})
    print(f"Train: {len(all_data['train'])} examples, {sum(all_data['train']['bytes']):,} bytes")
    print(f"Test ({FLORES_DATASET} {FLORES_SPLIT}): {len(all_data['test'])} examples, {sum(all_data['test']['bytes']):,} bytes")
    all_data.save_to_disk('data/translated_dataset')
    return all_data


def build_summarized_translated_dataset():
    en_data = load_dataset('json', data_files='data/summarized_translated/summaries.jsonl', split='train')
    en_nl = load_dataset('json', data_files='data/summarized_translated/dutch_translated_summaries.jsonl', split='train')
    en_zh = load_dataset('json', data_files='data/summarized_translated/chinese_translated_summaries.jsonl', split='train')

    text_bytes = total_column_bytes(en_data, 'text')
    summary_bytes = total_column_bytes(en_data, 'summary')
    dutch_bytes = total_column_bytes(en_nl, 'dutch')
    chinese_bytes = total_column_bytes(en_zh, 'chinese')
    en_bytes_nl = total_column_bytes(en_nl, 'text')
    en_bytes_zh = total_column_bytes(en_zh, 'text')
    if en_bytes_nl != en_bytes_zh:
        print(f"Warning: en summary bytes differ between en_nl ({en_bytes_nl:,}) and en_zh ({en_bytes_zh:,})")

    print("Byte counts (summarized dataset):")
    print(f"en_data text: {text_bytes:,}")
    print(f"en_data summary: {summary_bytes:,} ({summary_bytes / text_bytes:.4f} of text)")
    print(f"en_nl dutch: {dutch_bytes:,}")
    print(f"en_zh chinese: {chinese_bytes:,}")

    print("Byte counts (translations vs original English text):")
    check_byte_ratio('summary/text', summary_bytes, text_bytes, 1.0)
    check_byte_ratio('nl/en', dutch_bytes, text_bytes, byte_premiums['nld'])
    check_byte_ratio('zh/en', chinese_bytes, text_bytes, byte_premiums['zho'])

    en_nl = en_nl.rename_column('text', 'src').rename_column('dutch', 'tgt')
    en_zh = en_zh.rename_column('text', 'src').rename_column('chinese', 'tgt')

    data = concat_en_nl_en_zh(en_nl, en_zh)

    train = data.map(
        concat_sentences,
        batched=True,
        fn_kwargs={'lang1': 'english', 'lang2': 'dutch', 'lang3': 'chinese'},
        remove_columns=data.column_names,
        num_proc=20,
    )
    train = train.map(count_bytes, batched=True, num_proc=20)

    train_bytes = sum(train['bytes'])
    remaining = eng_100M_bytes - train_bytes
    print(f"Remaining bytes: {remaining:,}")
    if remaining > 0:
        filler = en_data.remove_columns(['summary']).map(count_bytes, batched=True, num_proc=20)
        filler = filter_to_byte_budget(filler, remaining)
        print(f"Adding {len(filler):,} original text examples ({sum(filler['bytes']):,} bytes) to reach {eng_100M_bytes:,} byte budget")
        train = concatenate_datasets([train, filler])

    test = build_flores_test()

    all_data = DatasetDict({'train': train, 'test': test})
    print(f"Train: {len(all_data['train'])} examples, {sum(all_data['train']['bytes']):,} bytes")
    print(f"Test ({FLORES_DATASET} {FLORES_SPLIT}): {len(all_data['test'])} examples, {sum(all_data['test']['bytes']):,} bytes")
    all_data.save_to_disk('data/summarized_dataset')
    return all_data

def build_summarized_dataset():
    english = load_dataset('json', data_files='data/summarized_translated/summaries.jsonl', split='train')
    dutch = load_dataset('json', data_files='data/summaries_nl.jsonl', split='train')
    chinese = load_dataset('json', data_files='data/summaries_zh.jsonl', split='train')

    english_bytes = total_column_bytes(english, 'summary')
    dutch_bytes = total_column_bytes(dutch, 'summary')
    chinese_bytes = total_column_bytes(chinese, 'summary')

    print("Byte counts (summarized dataset):")
    print(f"english summary: {english_bytes:,}")
    print(f"dutch summary: {dutch_bytes:,}")
    print(f"chinese summary: {chinese_bytes:,}")

    en_summary = english.remove_columns(['text'])
    dutch = dutch.remove_columns(['text'])
    chinese = chinese.remove_columns(['text'])


    train = concatenate_datasets([en_summary, dutch, chinese])
    train = train.rename_column('summary', 'text')    
    train = train.map(count_bytes, batched=True, num_proc=20)

    train_bytes = sum(train['bytes'])
    remaining = eng_100M_bytes - train_bytes
    print(f"Remaining bytes: {remaining:,}")
    if remaining > 0:
        filler = english.remove_columns(['summary']).map(count_bytes, batched=True, num_proc=20)
        filler = filter_to_byte_budget(filler, remaining)
        print(f"Adding {len(filler):,} original text examples ({sum(filler['bytes']):,} bytes) to reach {eng_100M_bytes:,} byte budget")
        train = concatenate_datasets([train, filler])

    test = build_flores_test()

    all_data = DatasetDict({'train': train, 'test': test})
    print(f"Train: {len(all_data['train'])} examples, {sum(all_data['train']['bytes']):,} bytes")
    print(f"Test ({FLORES_DATASET} {FLORES_SPLIT}): {len(all_data['test'])} examples, {sum(all_data['test']['bytes']):,} bytes")
    all_data.save_to_disk('data/summarized_dataset')
    return all_data

def build_packed_dataset(chunk_size=512):
    zho_data = load_dataset('BabyLM-community/babylm-zho', split='train')
    nld_data = load_dataset('BabyLM-community/babylm-nld', split='train')
    eng_data = load_dataset('BabyLM-community/BabyLM-2026-Strict', split='train')

    zho_data = zho_data.map(count_bytes, batched=True, num_proc=20)
    nld_data = nld_data.map(count_bytes, batched=True, num_proc=20)
    eng_data = eng_data.map(count_bytes, batched=True, num_proc=20)

    zho_data = filter_to_byte_budget(zho_data, zho_33M_bytes)
    nld_data = filter_to_byte_budget(nld_data, nld_33M_bytes)
    eng_data = filter_to_byte_budget(eng_data, eng_33M_bytes)

    zho_packed = Dataset.from_dict({'text': pack_texts(zho_data['text'], chunk_size)})
    nld_packed = Dataset.from_dict({'text': pack_texts(nld_data['text'], chunk_size)})
    eng_packed = Dataset.from_dict({'text': pack_texts(eng_data['text'], chunk_size)})

    zho_packed = zho_packed.map(count_bytes, batched=True, num_proc=20)
    nld_packed = nld_packed.map(count_bytes, batched=True, num_proc=20)
    eng_packed = eng_packed.map(count_bytes, batched=True, num_proc=20)

    train = concatenate_datasets([zho_packed, nld_packed, eng_packed])
    test = build_flores_test()

    all_data = DatasetDict({'train': train, 'test': test})
    print(f"Train: {len(all_data['train'])} examples, {sum(all_data['train']['bytes']):,} bytes")
    print(f"Test ({FLORES_DATASET} {FLORES_SPLIT}): {len(all_data['test'])} examples, {sum(all_data['test']['bytes']):,} bytes")
    all_data.save_to_disk('data/packed_dataset')
    return all_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='baseline')
    args = parser.parse_args()

    if args.dataset == 'baseline':
        build_baseline_dataset()
    elif args.dataset == 'translated':
        build_translated_dataset()
    elif args.dataset == 'packed_translated':
        build_packed_translated_dataset()
    elif args.dataset == 'summarized_translated':
        build_summarized_translated_dataset()
    elif args.dataset == 'summarized':
        build_summarized_dataset()
    elif args.dataset == 'packed':
        build_packed_dataset()
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")