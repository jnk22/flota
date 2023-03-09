import argparse
import pickle
from pathlib import Path

import pandas as pd
from data_helpers import ClassificationCollator, ClassificationDataset
from flota import FlotaTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

RUN_TYPES = ["train", "dev", "test"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=False, type=str, help="FLOTA mode")
    parser.add_argument(
        "--model", default=None, type=str, required=True, help="Name of model."
    )
    parser.add_argument(
        "--data", default=None, type=str, required=True, help="Name of data."
    )
    parser.add_argument(
        "--batch_size", default=None, type=int, required=True, help="Batch size."
    )
    parser.add_argument(
        "--strict", default=False, action="store_true", help="Use strict FLOTA."
    )
    parser.add_argument(
        "--k", default=None, type=int, required=False, help="Number of subwords."
    )
    parser.add_argument(
        "--limit_calls", default=0, type=int, help="Limit number of call/batch samples."
    )
    parser.add_argument(
        "--limit_encode", default=0, type=int, help="Limit number of encode samples."
    )
    parser.add_argument(
        "--limit_tokenize",
        default=0,
        type=int,
        help="Limit number of tokenize samples.",
    )
    parser.add_argument(
        "--output_dir",
        default="test_data",
        type=Path,
        help="Test data output directory.",
    )
    args = parser.parse_args()

    if not args.model.startswith("bert") and args.strict:
        print("Strict mode is only supported for BERT models.")
        exit(1)

    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    print(f"K: {args.k}")
    print(f"Strict: {args.strict}")
    print(f"Data: {args.data}")
    print(f"Batch size: {args.batch_size:02d}")

    tok = FlotaTokenizer(args.model, args.k, args.strict, args.mode)
    collator = ClassificationCollator(tok, False, 0)
    datasets = {
        rt: ClassificationDataset(pd.read_csv(f"data/{args.data}_{rt}.csv"))
        for rt in RUN_TYPES
    }
    loaders = {
        rt: DataLoader(datasets[rt], batch_size=args.batch_size, collate_fn=collator)
        for rt in RUN_TYPES
    }

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=datasets["train"].n_classes
    )
    if args.model == "gpt2":
        model.config.pad_token_id = model.config.eos_token_id

    # Run through data loaders to implicitly run FLOTA tokenization.
    for loader in loaders.values():
        for _ in loader:
            ...

    limits = (args.limit_tokenize, args.limit_encode, args.limit_calls)
    data_name = args.data.split("_")[1]

    for func_type, limit in zip(["tokenize", "encode", "call"], limits):
        strict_ext = "_strict" if args.strict else ""
        output_filename = f"{args.model}_{args.mode}_{args.k}{strict_ext}_{data_name}"

        files = {
            "input": args.output_dir / "input" / func_type / data_name,
            "output": args.output_dir / "output" / func_type / output_filename,
        }

        for io_type in ("input", "output"):
            files[io_type].parent.mkdir(parents=True, exist_ok=True)
            data = tok.test_data[io_type][func_type]
            data = data[:limit] if limit else data

            if isinstance(data, list) and isinstance(data[0], str):
                data = (f"{x}\n" for x in data)
                files[io_type].with_suffix(".txt").open("w").writelines(data)
            else:
                pickle.dump(data, files[io_type].with_suffix(".pkl").open("wb"))

    print("Cache info:")
    print(f"encode: {tok.encode.cache_info()}")
    print(f"tokenize: {tok.tokenize.cache_info()}")


if __name__ == "__main__":
    main()
