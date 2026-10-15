"""Deterministic group allocation with reserved label coverage."""
from collections import defaultdict
import hashlib
import json
import time

import numpy as np

SPLITS = ("train", "validation", "test")
RATIOS = np.array([0.70, 0.15, 0.15])


def _tie_hash(seed, tag, group_id):
    value = f"{seed}\0{tag}\0{group_id}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _digest(value):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _feature_counts(samples, group_ids):
    features = []
    weights = []
    projects = set()
    for sample in samples:
        projects.add(sample["project_id"])
    for project in sorted(projects):
        observed_labels = set()
        core_labels = set()
        for sample in samples:
            if sample["project_id"] != project:
                continue
            observed_labels.update(sample["labels_observed"])
            if sample["core_eligible"]:
                core_labels.add(sample["labels_observed"][0])
        for label in sorted(core_labels):
            features.append(("core", project, label))
            weights.append(1.0)
        for label in sorted(observed_labels):
            features.append(("observed", project, label))
            weights.append(0.25)
        features.extend([("core_size", project, ""), ("all_size", project, "")])
        weights.extend([2.0, 0.25])
    group_index = {group: index for index, group in enumerate(group_ids)}
    feature_index = {feature: index for index, feature in enumerate(features)}
    counts = np.zeros((len(group_ids), len(features)), dtype=np.int64)
    for sample in samples:
        group = group_index[sample["leakage_group_id"]]
        project = sample["project_id"]
        counts[group, feature_index[("all_size", project, "")]] += 1
        for label in sample["labels_observed"]:
            counts[group, feature_index[("observed", project, label)]] += 1
        if sample["core_eligible"]:
            label = sample["labels_observed"][0]
            counts[group, feature_index[("core", project, label)]] += 1
            counts[group, feature_index[("core_size", project, "")]] += 1
    return features, weights, counts


class _CoverageAllocator:
    def __init__(self, samples, groups, seed):
        self.seed = seed
        self.ids = [group["leakage_group_id"] for group in groups]
        self.index = {group: index for index, group in enumerate(self.ids)}
        self.features, weights, self.matrix = _feature_counts(samples, self.ids)
        self.totals = self.matrix.sum(axis=0)
        self.group_support = (self.matrix > 0).sum(axis=0)
        self.weights = np.array(weights) / np.maximum(self.totals, 1)
        self.targets = RATIOS[:, None] * self.totals[None, :]
        self.assigned = np.full(len(self.ids), -1, dtype=np.int8)
        self.reasons = defaultdict(list)
        self.label_features = []
        for index, feature in enumerate(self.features):
            if feature[0] in ("core", "observed"):
                self.label_features.append(index)

    def _hash(self, tag, group):
        return _tie_hash(self.seed, tag, self.ids[group])

    def _feature_order(self, feature):
        return self.group_support[feature], self.features[feature]

    def _has_positive(self, split, feature):
        return np.any(self.matrix[self.assigned == split, feature] > 0)

    def _missing_features(self, split, features):
        missing = set()
        for feature in features:
            if not self._has_positive(split, feature):
                missing.add(feature)
        return missing

    def _reserve(self, group, split, reason, feature=None):
        self.assigned[group] = split
        details = {"reason": reason}
        if feature is not None:
            details["project_id"] = self.features[feature][1]
            details["label"] = self.features[feature][2]
        self.reasons[self.ids[group]].append(details)

    def reserve_training(self, forced_groups):
        for group_id in forced_groups:
            self._reserve(self.index[group_id], 0, "common_scope_train_only_group")
        for feature, description in enumerate(self.features):
            if description[0] != "core" or self.group_support[feature] > 2:
                continue
            for group in np.flatnonzero(self.matrix[:, feature]):
                self._reserve(group, 0, "core_label_at_most_two_groups", feature)
        for feature in sorted(self.label_features, key=self._feature_order):
            if self._has_positive(0, feature):
                continue
            missing = self._missing_features(0, self.label_features)
            choices = np.flatnonzero(self.matrix[:, feature] > 0).tolist()
            scores = []
            for group in choices:
                covered = 0
                for label_feature in missing:
                    covered += self.matrix[group, label_feature] > 0
                score = (-covered, int(self.matrix[group].sum()),
                         self._hash("reserve", group), group)
                scores.append(score)
            chosen = min(scores)[-1]
            self._reserve(chosen, 0, "reserve_train_label_coverage", feature)

    def _holdout_score(self, group, split, core_features, missing):
        other_split = 3 - split
        blocked = 0
        for feature in core_features:
            if not self.matrix[group, feature]:
                continue
            if self._has_positive(other_split, feature):
                continue
            available = (self.matrix[:, feature] > 0) & (self.assigned < 0)
            if int(np.sum(available)) <= 1:
                blocked += 1
        coverage = 0
        for feature in missing:
            coverage += (self.matrix[group, feature] > 0) / max(
                int(self.group_support[feature]), 1)
        return (blocked, -coverage, int(self.matrix[group].sum()),
                self._hash("reserve" + str(split), group))

    def reserve_holdouts(self, holdout_labels):
        core_features = []
        for feature, description in enumerate(self.features):
            if description[0] != "core" or self.group_support[feature] < 3:
                continue
            if holdout_labels is not None and description[2] not in holdout_labels:
                continue
            core_features.append(feature)
        for feature in sorted(core_features, key=self._feature_order):
            for split in (1, 2):
                if self._has_positive(split, feature):
                    continue
                available = (self.matrix[:, feature] > 0) & (self.assigned < 0)
                choices = np.flatnonzero(available).tolist()
                if not choices:
                    raise ValueError(
                        f"Cannot reserve holdout coverage: {self.features[feature]}, split {split}")
                missing = self._missing_features(split, core_features)
                scored = []
                for group in choices:
                    score = self._holdout_score(group, split, core_features, missing)
                    scored.append((score, group))
                score, chosen = min(scored)
                if score[0]:
                    raise ValueError(
                        f"Coupled group constraints block a holdout: {self.features[feature]}")
                reason = "reserve_" + SPLITS[split] + "_label_coverage"
                self._reserve(chosen, split, reason, feature)

    def fill_unassigned(self):
        self.counts = np.zeros((3, len(self.features)), dtype=np.int64)
        for split in range(3):
            self.counts[split] = self.matrix[self.assigned == split].sum(axis=0)
        rarity = (self.matrix / np.maximum(self.totals, 1)).sum(axis=1)
        order = []
        for group in np.flatnonzero(self.assigned < 0).tolist():
            order.append((-float(rarity[group]), -int(self.matrix[group].sum()),
                          self._hash("order", group), group))
        for _, _, _, group in sorted(order):
            row = self.matrix[group]
            changes = ((2 * (self.counts - self.targets) * row + row * row)
                       * self.weights).sum(axis=1)
            options = []
            for split in range(3):
                options.append((round(float(changes[split]), 12),
                                self._hash("split" + str(split), group), split))
            chosen = min(options)[-1]
            self.assigned[group] = chosen
            self.counts[chosen] += row

    def rebalance(self):
        order = []
        for group, group_id in enumerate(self.ids):
            if group_id not in self.reasons:
                order.append((self._hash("rebalance", group), group))
        order.sort()
        moves_per_sweep = []
        for _ in range(30):
            moved = 0
            for _, group in order:
                old_split = int(self.assigned[group])
                row = self.matrix[group]
                removal = ((-2 * (self.counts[old_split] - self.targets[old_split])
                            * row + row * row) * self.weights).sum()
                options = []
                for split in range(3):
                    if split == old_split:
                        continue
                    addition = ((2 * (self.counts[split] - self.targets[split])
                                 * row + row * row) * self.weights).sum()
                    change = float(removal + addition)
                    options.append((round(change, 12),
                                    self._hash("move" + str(split), group), split))
                change, _, chosen = min(options)
                if change < -1e-9:
                    self.counts[old_split] -= row
                    self.counts[chosen] += row
                    self.assigned[group] = chosen
                    moved += 1
            moves_per_sweep.append(moved)
            if moved == 0:
                break
        return moves_per_sweep

    def finish(self, samples, groups, moves_per_sweep):
        split_by_group = {}
        for index, group_id in enumerate(self.ids):
            split_by_group[group_id] = SPLITS[int(self.assigned[index])]
        for sample in samples:
            sample["split"] = split_by_group[sample["leakage_group_id"]]
        for group in groups:
            group["split"] = split_by_group[group["leakage_group_id"]]
        missing = []
        for feature in self.label_features:
            if self.counts[0, feature] == 0:
                missing.append(self.features[feature])
        if missing:
            raise ValueError(f"Training label coverage failed: {missing}")
        reserved = []
        for group_id in sorted(self.reasons):
            reserved.append({"leakage_group_id": group_id,
                             "split": split_by_group[group_id],
                             "reasons": self.reasons[group_id]})
        objective = float((((self.counts - self.targets) ** 2) * self.weights).sum())
        return {"algorithm": "coverage_reserved_greedy", "seed": self.seed,
                "requested_ratios": dict(zip(SPLITS, RATIOS.tolist())),
                "objective_formula": "sum_over_splits_and_features(weight[feature] * (actual - ratio * total)^2 / max(total, 1))",
                "objective_value": objective, "reserved_groups": reserved,
                "balance_sweep_move_counts": moves_per_sweep,
                "model_scores_used": False, "official_split": False}


def allocate_splits(samples, groups, forced_train_groups=(), holdout_labels=None, seed=10517):
    """Mutate sample/group splits; keep coverage reservations fixed while balancing."""
    allocator = _CoverageAllocator(samples, groups, seed)
    allocator.reserve_training(forced_train_groups)
    allocator.reserve_holdouts(holdout_labels)
    allocator.fill_unassigned()
    moves = allocator.rebalance()
    return allocator.finish(samples, groups, moves)


class _LinearConstraints:
    def __init__(self):
        self.rows = []
        self.columns = []
        self.values = []
        self.lower = []
        self.upper = []

    def add(self, terms, lower, upper):
        row = len(self.lower)
        self.lower.append(lower)
        self.upper.append(upper)
        for column, value in terms:
            if value:
                self.rows.append(row)
                self.columns.append(column)
                self.values.append(value)

    def matrix(self, num_variables):
        from scipy.sparse import coo_matrix
        shape = (len(self.lower), num_variables)
        return coo_matrix((self.values, (self.rows, self.columns)), shape=shape).tocsc()


def _milp_data(samples, seed):
    grouped = defaultdict(list)
    labels = set()
    for sample in samples:
        grouped[sample["leakage_group_id"]].append(sample)
        labels.add(sample["labels_observed"][0])
    order = []
    for group_id in grouped:
        order.append((_digest([seed, "milp_column", group_id]), group_id))
    group_ids = [group_id for _, group_id in sorted(order)]
    labels = sorted(labels)
    counts = []
    for group_id in group_ids:
        row = []
        for label in labels:
            positives = 0
            for sample in grouped[group_id]:
                positives += sample["labels_observed"][0] == label
            row.append(positives)
        row.append(len(grouped[group_id]))
        counts.append(row)
    return group_ids, labels, np.array(counts, dtype=float)


def _milp_problem(group_ids, labels, counts, forced, evaluation_labels, seed):
    num_groups, num_features = counts.shape
    assignment_variables = num_groups * 3
    num_variables = assignment_variables + 3 * num_features
    objective = np.zeros(num_variables)
    for group, group_id in enumerate(group_ids):
        for split in range(3):
            tie_value = int(_digest([seed, group_id, split])[:8], 16) / 2 ** 32
            objective[3 * group + split] = 1e-10 * tie_value
    feature_weights = np.array([1.25] * len(labels) + [2.25])
    for split in range(3):
        start = assignment_variables + split * num_features
        objective[start:start + num_features] = feature_weights / np.maximum(counts.sum(0), 1)
    lower = np.zeros(num_variables)
    upper = np.full(num_variables, np.inf)
    upper[:assignment_variables] = 1
    for group, group_id in enumerate(group_ids):
        if group_id in forced:
            lower[3 * group] = 1
            upper[3 * group] = 1
            upper[3 * group + 1] = 0
            upper[3 * group + 2] = 0
    constraints = _LinearConstraints()
    for group in range(num_groups):
        terms = [(3 * group + split, 1) for split in range(3)]
        constraints.add(terms, 1, 1)
    for feature, label in enumerate(labels):
        terms = []
        for group in range(num_groups):
            terms.append((3 * group, float(counts[group, feature] > 0)))
        constraints.add(terms, 1, np.inf)
        if label in evaluation_labels:
            for split in (1, 2):
                terms = []
                for group in range(num_groups):
                    terms.append((3 * group + split, float(counts[group, feature] > 0)))
                constraints.add(terms, 1, np.inf)
    targets = RATIOS[:, None] * counts.sum(0)
    for split in range(3):
        for feature in range(num_features):
            slack = assignment_variables + split * num_features + feature
            terms = []
            inverse_terms = []
            for group in range(num_groups):
                column = 3 * group + split
                value = counts[group, feature]
                terms.append((column, value))
                inverse_terms.append((column, -value))
            constraints.add(terms + [(slack, -1)], -np.inf, targets[split, feature])
            constraints.add(inverse_terms + [(slack, -1)], -np.inf, -targets[split, feature])
    return objective, lower, upper, constraints


def _check_milp_splits(samples, split_by_group, forced, labels, evaluation_labels):
    for group_id in forced:
        if split_by_group[group_id] != "train":
            raise ValueError("MILP moved a forced training group")
    positive_labels = defaultdict(set)
    for sample in samples:
        sample["split"] = split_by_group[sample["leakage_group_id"]]
        positive_labels[sample["split"]].add(sample["labels_observed"][0])
    for label in labels:
        if label not in positive_labels["train"]:
            raise ValueError("MILP training label coverage failed")
        if label in evaluation_labels:
            for split in ("validation", "test"):
                if label not in positive_labels[split]:
                    raise ValueError("MILP holdout label coverage failed")


def fallback_milp(samples, forced, evaluation_labels, seed):
    """Use a 30-second MILP fallback with weighted absolute count deviation."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    group_ids, labels, counts = _milp_data(samples, seed)
    objective, lower, upper, constraints = _milp_problem(
        group_ids, labels, counts, forced, evaluation_labels, seed)
    assignment_variables = len(group_ids) * 3
    integrality = np.r_[np.ones(assignment_variables), np.zeros(3 * counts.shape[1])]
    matrix = constraints.matrix(len(objective))
    start = time.perf_counter()
    solved = milp(
        objective, integrality=integrality, bounds=Bounds(lower, upper),
        constraints=LinearConstraint(matrix, np.array(constraints.lower),
                                     np.array(constraints.upper)),
        options={"time_limit": 30.0, "mip_rel_gap": 0.01})
    audit = {"algorithm": "bounded_MILP_L1_balance_fallback", "seed": seed,
             "status": int(solved.status), "message": str(solved.message),
             "seconds": time.perf_counter() - start,
             "objective": "weighted absolute count deviation; not the original squared greedy objective",
             "model_scores_used": False}
    if solved.x is None:
        raise RuntimeError("MILP has no validated feasible witness: " + json.dumps(audit))
    binary = np.rint(solved.x[:assignment_variables]).astype(int).reshape(len(group_ids), 3)
    if not np.all(binary.sum(1) == 1):
        raise ValueError("MILP group assignment is not one-hot")
    split_by_group = {}
    for index, group_id in enumerate(group_ids):
        split_by_group[group_id] = SPLITS[int(binary[index].argmax())]
    _check_milp_splits(samples, split_by_group, forced, labels, evaluation_labels)
    audit["feasible_witness_independently_checked"] = True
    audit["objective_value"] = float(solved.fun)
    return audit
