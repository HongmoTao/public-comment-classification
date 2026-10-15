"""Score frozen predictions and count the document groups they cover.

Metrics use a 0--100 scale. Nested evaluation pools the unique outer
predictions and keeps every evaluation label in the macro-F1 denominator.
"""
from collections import defaultdict

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from common.models import METHODS, macro_f1


def check_labels(labels, k):
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("Evaluation labels must be nonempty and unique")
    for label in labels:
        if not isinstance(label, (int, np.integer)) or not 0 <= label < k:
            raise ValueError("Evaluation label is outside the output range")


def identity_for_row(row):
    values = []
    for field in ("sample_id", "fold", "reference_label_id", "group_id"):
        values.append(row[field])
    return tuple(values)


def validate_predictions(data):
    k = data["num_labels"]
    if not isinstance(k, int) or k < 1:
        raise ValueError("num_labels must be a positive integer")
    check_labels(data["evaluation_labels"], k)
    predictions = data["predictions"]
    anchor = predictions[METHODS[0]]
    if not anchor:
        raise ValueError("Test predictions must be nonempty")
    expected = [identity_for_row(row) for row in anchor]
    seen = set()
    identity = {}
    for row in anchor:
        key = (row["sample_id"], row["fold"])
        if key in seen:
            raise ValueError("A test fold contains repeated sample IDs")
        seen.add(key)
        value = (row["reference_label_id"], row["group_id"])
        previous = identity.setdefault(row["sample_id"], value)
        if previous != value:
            raise ValueError("Repeated samples have inconsistent truth or groups")
    for method in METHODS:
        rows = predictions[method]
        actual = [identity_for_row(row) for row in rows]
        if actual != expected:
            raise ValueError("Representations must predict the same ordered test rows")
        for row in rows:
            for field in ("reference_label_id", "predicted_label_id"):
                value = row[field]
                if not isinstance(value, (int, np.integer)) or not 0 <= value < k:
                    raise ValueError("Prediction or reference label is outside the output range")
    if data["mode"] == "nested" and len(identity) != len(anchor):
        raise ValueError("Nested outer predictions contain repeated sample IDs")
    return anchor, sorted({row["fold"] for row in anchor})


def summarize(rows, labels, k):
    check_labels(labels, k)
    truth = [row["reference_label_id"] for row in rows]
    predicted = [row["predicted_label_id"] for row in rows]
    grouped = defaultdict(list)
    for row in rows:
        correct = row["reference_label_id"] == row["predicted_label_id"]
        grouped[row["group_id"]].append(correct)
    accuracy = 0.0
    group_accuracy = 0.0
    score = 0.0
    f1 = np.zeros(k)
    support = np.zeros(k, dtype=int)
    confusion = np.zeros((k, k), dtype=int)
    if rows:
        accuracy = 100 * float(np.mean(np.asarray(truth) == predicted))
        group_scores = []
        for correctness in grouped.values():
            group_scores.append(float(np.mean(correctness)))
        group_accuracy = 100 * float(np.mean(group_scores))
        score = 100 * macro_f1(truth, predicted, labels)
        _, _, f1, support = precision_recall_fscore_support(
            truth, predicted, labels=list(range(k)), zero_division=0)
        confusion = confusion_matrix(truth, predicted, labels=list(range(k)))
    return {"macro_f1": score,
            "accuracy": accuracy, "group_accuracy": group_accuracy,
            "texts": len(rows), "groups": len(grouped),
            "confusion": confusion.tolist(),
            "class_f1": (100 * f1).tolist(), "class_support": support.tolist()}


def fold_labels(data, fold):
    labels = data["evaluation_labels"]
    if data["mode"] != "nested":
        by_fold = data.get("test_labels", {})
        labels = by_fold.get(str(fold), labels)
    check_labels(labels, data["num_labels"])
    return labels


def summarize_folds(data, folds):
    result = {}
    for fold in folds:
        scores = {}
        labels = fold_labels(data, fold)
        for method in METHODS:
            rows = [row for row in data["predictions"][method] if row["fold"] == fold]
            scores[method] = summarize(rows, labels, data["num_labels"])
        result[str(fold)] = scores
    return result


def mean_holdouts(per_fold):
    result = {}
    for method in METHODS:
        averages = {}
        for metric in ("macro_f1", "accuracy", "group_accuracy"):
            values = [scores[method][metric] for scores in per_fold.values()]
            averages[metric] = float(np.mean(values))
        result[method] = averages
    return result


def historical_coverage(plan, rows, labels):
    """Count partition membership; class counts use available prediction truth."""
    by_id = {row["sample_id"]: row for row in rows}
    source = plan["sample_to_source_group"]
    cumulative = set()
    missing = set()
    positives = defaultdict(set)
    fold_records = []
    for index, fold in enumerate(plan["folds"]):
        ids = fold["test"]
        if len(ids) != len(set(ids)):
            raise ValueError("Historical test partition contains repeated sample IDs")
        groups = {source[sid] for sid in ids}
        new_groups = groups - cumulative
        cumulative.update(groups)
        fold_positive = defaultdict(set)
        unknown = []
        for sid in ids:
            if sid not in by_id:
                missing.add(sid)
                unknown.append(sid)
                continue
            row = by_id[sid]
            if row["group_id"] != source[sid]:
                raise ValueError("Historical and current source groups differ")
            label = row["reference_label_id"]
            fold_positive[label].add(source[sid])
            positives[label].add(source[sid])
        classes = []
        for label in labels:
            classes.append({"label_id": label,
                            "positive_groups": len(fold_positive[label]),
                            "positive_group_ids": sorted(fold_positive[label])})
        fold_records.append({"fold": index, "texts": len(ids), "groups": len(groups),
                             "group_ids": sorted(groups), "new_groups": len(new_groups),
                             "cumulative_distinct_groups": len(cumulative),
                             "missing_truth_texts": len(unknown), "class_coverage": classes})
    classes = []
    for label in labels:
        classes.append({"label_id": label, "positive_groups": len(positives[label]),
                        "positive_group_ids": sorted(positives[label])})
    result = {"per_fold": fold_records, "distinct_test_groups": len(cumulative),
              "missing_truth_texts": len(missing), "class_coverage": classes,
              "class_truth_scope": "Samples present in the current predictions"}
    return result, cumulative, positives


def class_coverage(rows, labels, earlier_positive=None):
    result = []
    for label in labels:
        positive = [row for row in rows if row["reference_label_id"] == label]
        ids = {row["sample_id"] for row in positive}
        groups = {row["group_id"] for row in positive}
        record = {"label_id": label, "positive_texts": len(ids),
                  "positive_groups": len(groups)}
        if earlier_positive is not None:
            record["earlier_positive_groups"] = len(groups & earlier_positive[label])
        result.append(record)
    return result


def class_contributions(pooled, labels):
    result = []
    for label in labels:
        record = {"label_id": label}
        combined = pooled["combined"]["class_f1"][label]
        for baseline in ("lexical", "semantic"):
            difference = combined - pooled[baseline]["class_f1"][label]
            record["combined_minus_" + baseline + "_pp"] = difference / len(labels)
        result.append(record)
    return result


def paired_decisions(predictions):
    result = []
    correct = {}
    for method in METHODS:
        correct[method] = np.array([
            row["reference_label_id"] == row["predicted_label_id"]
            for row in predictions[method]])
    a = correct["combined"]
    for baseline in ("lexical", "semantic"):
        b = correct[baseline]
        result.append({"baseline": baseline, "both_correct": int((a & b).sum()),
                       "combined_only": int((a & ~b).sum()),
                       "baseline_only": int((~a & b).sum()),
                       "both_wrong": int((~a & ~b).sum())})
    return result


def historical_subsets(data, earlier):
    result = {}
    for name, seen in (("earlier", True), ("previously_untested", False)):
        scores = {}
        for method in METHODS:
            rows = []
            for row in data["predictions"][method]:
                if (row["group_id"] in earlier) == seen:
                    rows.append(row)
            scores[method] = summarize(rows, data["evaluation_labels"], data["num_labels"])
        result[name] = scores
    return result


def analyze(data, historical=None):
    anchor, folds = validate_predictions(data)
    labels = data["evaluation_labels"]
    per_fold = summarize_folds(data, folds)
    result = {"mode": data["mode"], "per_fold": per_fold,
              "distinct_test_texts": len({row["sample_id"] for row in anchor}),
              "distinct_test_groups": len({row["group_id"] for row in anchor})}
    if data["mode"] == "nested":
        pooled = {}
        for method in METHODS:
            pooled[method] = summarize(data["predictions"][method], labels, data["num_labels"])
        result["pooled"] = pooled
        result["class_contributions"] = class_contributions(pooled, labels)
        result["paired_decisions"] = paired_decisions(data["predictions"])
    else:
        result["mean_holdout"] = mean_holdouts(per_fold)
        if len(folds) == 1:
            result["summary"] = per_fold[str(folds[0])]
    earlier_positive = None
    if historical is not None:
        coverage, earlier, earlier_positive = historical_coverage(historical, anchor, labels)
        result["historical_coverage"] = coverage
        if data["mode"] == "nested":
            result["historical_subsets"] = historical_subsets(data, earlier)
    result["class_coverage"] = class_coverage(anchor, labels, earlier_positive)
    return result
