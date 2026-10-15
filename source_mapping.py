"""Build source components from every canonical sample before pool filtering.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from common.io import read_rows, write_json


def _document_bodies(documents):
    bodies = []
    document_keys = set()
    for document in documents:
        project = document.get("project_id")
        document_id = document["document_id"]
        key = (project, document_id)
        if key in document_keys:
            raise ValueError("Repeated document_id within a project")
        document_keys.add(key)
        if "pages" in document:
            pages = document["pages"]
        else:
            pages = [document.get("text")]
        if not isinstance(pages, list):
            raise ValueError("Document pages must be a list of strings")
        for page in pages:
            if not isinstance(page, str):
                raise ValueError("Documents need text or a list of page strings")
        body = "\n".join(pages)
        bodies.append({"project_id": project, "document_id": document_id,
                       "text": body, "normalized_text": " ".join(body.split())})
    return bodies


def _source_candidates(sample, bodies):
    text = sample["text"]
    normalized_text = " ".join(text.split())
    if not normalized_text or not sample["leakage_group_id"]:
        raise ValueError("Empty sample text or leakage_group_id")
    candidates = set()
    for document in bodies:
        project = document["project_id"]
        if project is not None and project != sample.get("project_id"):
            continue
        literal_match = text in document["text"]
        whitespace_match = normalized_text in document["normalized_text"]
        if literal_match or whitespace_match:
            candidates.add(document["document_id"])
    return sorted(candidates)


class _Components:
    def __init__(self, sample_ids):
        self.parent = {sample_id: sample_id for sample_id in sample_ids}
        self.seen_relations = {}

    def find(self, sample_id):
        while self.parent[sample_id] != sample_id:
            self.parent[sample_id] = self.parent[self.parent[sample_id]]
            sample_id = self.parent[sample_id]
        return sample_id

    def add_relation(self, sample_id, relation):
        if relation not in self.seen_relations:
            self.seen_relations[relation] = sample_id
            return
        left = self.find(sample_id)
        right = self.find(self.seen_relations[relation])
        if left != right:
            self.parent[max(left, right)] = min(left, right)

    def groups(self):
        components = defaultdict(list)
        for sample_id in self.parent:
            components[self.find(sample_id)].append(sample_id)
        group_for = {}
        for members in components.values():
            text = json.dumps(sorted(members), ensure_ascii=False, sort_keys=True)
            group_id = "source-group-" + hashlib.sha256(text.encode("utf-8")).hexdigest()
            for sample_id in members:
                group_for[sample_id] = group_id
        return components, group_for


def _pool_ids(samples, pool, by_id):
    selected = samples
    if pool is not None:
        selected = pool
    ids = []
    for row in selected:
        sample_id = row["sample_id"]
        ids.append(sample_id)
        if sample_id not in by_id:
            raise ValueError("Pool contains an unknown sample_id")
        original = by_id[sample_id]
        if "project_id" in original and "project_id" in row:
            if row["project_id"] != original["project_id"]:
                raise ValueError("Pool project_id differs from canonical samples")
        if "text" in row and row["text"] != original["text"]:
            raise ValueError("Pool text differs from canonical samples")
    if len(set(ids)) != len(ids):
        raise ValueError("Pool contains repeated sample_id values")
    return sorted(ids)


def build_sources(samples, documents, pool=None):
    """Keep all matches and allow excluded samples to bridge source components."""
    by_id = {}
    for sample in samples:
        by_id[sample["sample_id"]] = sample
    if not samples or len(by_id) != len(samples):
        raise ValueError("Samples must be nonempty with unique sample_id values")
    bodies = _document_bodies(documents)
    graph = _Components(by_id)
    candidates = {}
    unmatched = 0
    for sample in samples:
        sample_id = sample["sample_id"]
        project = sample.get("project_id")
        matches = _source_candidates(sample, bodies)
        candidates[sample_id] = matches
        if not matches:
            unmatched += 1
        graph.add_relation(sample_id, (project, "text", sample["leakage_group_id"]))
        for document_id in matches:
            graph.add_relation(sample_id, (project, "document", document_id))
    if unmatched:
        raise ValueError(f"{unmatched} samples have no source candidates")
    components, group_for = graph.groups()
    selected_ids = _pool_ids(samples, pool, by_id)
    selected_groups = {}
    selected_candidates = {}
    occupied_groups = set()
    for sample_id in selected_ids:
        selected_groups[sample_id] = group_for[sample_id]
        selected_candidates[sample_id] = candidates[sample_id]
        occupied_groups.add(group_for[sample_id])
    statistics = {"full_samples": len(samples), "documents": len(bodies),
                  "full_components": len(components),
                  "largest_full_component": max(map(len, components.values())),
                  "pool_samples": len(selected_ids),
                  "pool_components": len(occupied_groups)}
    return {"sample_to_source_group": selected_groups,
            "sample_to_candidates": selected_candidates,
            "full_sample_to_source_group": dict(sorted(group_for.items())),
            "statistics": statistics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--pool", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    samples = read_rows(args.samples, key="samples")
    documents = read_rows(args.documents)
    pool = None
    if args.pool is not None:
        pool = read_rows(args.pool, key="rows")
    write_json(args.output, build_sources(samples, documents, pool))


if __name__ == "__main__":
    main()
