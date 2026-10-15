"""Encode sample_id/text rows with a frozen sentence model.

Example: python common/encoding.py --pool train.jsonl validation.jsonl test.jsonl
    --output embeddings.npz
Dependencies: torch, transformers, huggingface-hub, safetensors, numpy.
"""
import argparse
import os
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io import read_rows

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_embeddings(path, rows):
    with np.load(path, allow_pickle=False) as archive:
        sample_ids = archive["sample_ids"].tolist()
        embeddings = archive["embeddings"]
    if not isinstance(sample_ids, list):
        raise ValueError("Embedding IDs must be a vector")
    for sample_id in sample_ids:
        if not isinstance(sample_id, str):
            raise ValueError("Embedding IDs must be strings")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate embedding IDs")
    if embeddings.ndim != 2 or embeddings.shape[0] != len(sample_ids):
        raise ValueError("Embedding IDs and matrix shape disagree")
    if embeddings.shape[1] != 384:
        raise ValueError("Expected 384-dimensional sentence vectors")
    if embeddings.dtype != np.dtype(np.float32):
        raise ValueError("Embeddings must use float32")
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain nonfinite values")
    row_norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(row_norms, 1.0, atol=2e-4):
        raise ValueError("Embeddings must have unit length")

    cached_positions = {}
    for index, sample_id in enumerate(sample_ids):
        cached_positions[sample_id] = index
    positions = {}
    selected_indices = []
    for index, row in enumerate(rows):
        sample_id = row["sample_id"]
        if sample_id in positions:
            raise ValueError("Duplicate requested sample IDs")
        if sample_id not in cached_positions:
            raise ValueError("Missing embedding sample ID")
        positions[sample_id] = index
        selected_indices.append(cached_positions[sample_id])
    return positions, embeddings[selected_indices]


def load_model(args):
    import torch
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    cache_directory = args.output.parent / ".model_cache"
    snapshot_download(
        args.model,
        revision=args.revision,
        local_dir=cache_directory,
        local_dir_use_symlinks=False,
        token=False,
        allow_patterns=(
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
        ),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cache_directory,
        use_fast=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer.truncation_side = "right"
    model = AutoModel.from_pretrained(
        cache_directory,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        local_files_only=True,
        trust_remote_code=False,
    )
    model.eval()
    model.requires_grad_(False)
    model.to(args.device)
    return tokenizer, model


def encode_batch(rows, tokenizer, model, device):
    import torch

    texts = []
    for row in rows:
        texts.append(row["text"])
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    for name, tensor in batch.items():
        batch[name] = tensor.to(device)
    hidden = model(**batch).last_hidden_state.float()
    mask = batch["attention_mask"].unsqueeze(-1).float()
    token_counts = mask.sum(1)
    if (token_counts == 0).any():
        raise ValueError("All-padding input")
    pooled = (hidden * mask).sum(1) / token_counts
    if not torch.isfinite(pooled).all():
        raise ValueError("Pooled embeddings contain nonfinite values")
    if (pooled.norm(dim=1) < 1e-12).any():
        raise ValueError("Pooled embeddings have zero norm")
    normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return normalized.cpu().numpy()


def sample_order(row):
    return row["sample_id"]


def encoding_rows(paths):
    rows = []
    seen = set()
    for path in paths:
        records = read_rows(path)
        for row in records:
            sample_id = row["sample_id"]
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError("Expected a nonempty string sample ID")
            if sample_id in seen:
                raise ValueError("Repeated sample ID in encoding inputs")
            if not isinstance(row["text"], str) or not row["text"].strip():
                raise ValueError("Expected nonempty excerpt text")
            seen.add(sample_id)
            rows.append(row)
    if not rows:
        raise ValueError("Encoding inputs must not be empty")
    return sorted(rows, key=sample_order)


def encode(args):
    import torch

    rows = encoding_rows(args.pool)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer, model = load_model(args)
    embeddings = []
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start:start + args.batch_size]
            encoded = encode_batch(batch_rows, tokenizer, model, args.device)
            embeddings.append(encoded)
    sample_ids = []
    for row in rows:
        sample_ids.append(row["sample_id"])
    matrix = np.concatenate(embeddings).astype(np.float32)
    requested_revision = "default"
    if args.revision is not None:
        requested_revision = args.revision
    np.savez_compressed(
        args.output,
        sample_ids=np.asarray(sample_ids),
        embeddings=matrix,
        model=np.asarray(args.model),
        revision=np.asarray(requested_revision),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, nargs="+", required=True,
                        help="One pool file, or train/validation/test record files")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision",
                        help="Optional model branch, tag or commit; omitted uses the repository default")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    encode(args)


if __name__ == "__main__":
    main()
