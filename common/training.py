"""Shared fitting procedures; model inputs are excerpt text only.
Nested evaluation pools inner predictions, then refits on all outer training.
"""
import hashlib
import json
import math
import os

for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(variable, '4')

import numpy as np
from common.encoding import load_embeddings
from common.models import METHODS, CANDIDATES, make_features, combine
from common.models import fit_classifier, predict, macro_f1
from common.plans import validate_plan


def select_training_ids(ids, by_id, fraction, seed, num_labels):
    """Take a class-covered fraction of text groups, preserving input order."""
    if not 0 < fraction <= 1:
        raise ValueError('Training fraction must be in (0, 1]')
    per_label = {}
    for label in range(num_labels):
        per_label[label] = set()
    for sid in ids:
        row = by_id[sid]
        per_label[row['label_id']].add(row['leakage_group_id'])
    chosen = set()
    for label, groups in per_label.items():
        if not groups:
            raise ValueError('Full training partition lacks an output label')
        def group_order(group):
            value = [seed, label, group]
            text = json.dumps(value, sort_keys=True, ensure_ascii=False)
            return hashlib.sha256(text.encode()).hexdigest()
        ordered = sorted(groups, key=group_order)
        count = max(1, math.ceil(len(groups) * fraction))
        chosen.update(ordered[:count])
    selected = []
    for sid in ids:
        if by_id[sid]['leakage_group_id'] in chosen:
            selected.append(sid)
    return selected


def texts_for(ids, by_id):
    texts = []
    for sid in ids:
        texts.append(by_id[sid]['text'])
    return texts


def targets_for(ids, by_id):
    targets = []
    for sid in ids:
        targets.append(by_id[sid]['label_id'])
    return np.asarray(targets, dtype=np.int64)


def vectors_for(ids, positions, embeddings):
    indices = []
    for sid in ids:
        indices.append(positions[sid])
    return embeddings[indices]


def fit_features(train_ids, query_ids, by_id, positions, embeddings):
    vectorizer = make_features()
    training_texts = texts_for(train_ids, by_id)
    query_texts = texts_for(query_ids, by_id)
    lexical_train = vectorizer.fit_transform(training_texts).tocsr()
    lexical_query = vectorizer.transform(query_texts).tocsr()
    train_vectors = vectors_for(train_ids, positions, embeddings)
    query_vectors = vectors_for(query_ids, positions, embeddings)
    matrices = {}
    for method in METHODS:
        training = combine(method, lexical_train, train_vectors)
        query = combine(method, lexical_query, query_vectors)
        matrices[method] = (training, query)
    return vectorizer, matrices


def query_features(vectorizer, ids, by_id, positions, embeddings):
    texts = texts_for(ids, by_id)
    lexical = vectorizer.transform(texts).tocsr()
    vectors = vectors_for(ids, positions, embeddings)
    matrices = {}
    for method in METHODS:
        matrices[method] = combine(method, lexical, vectors)
    return matrices


def candidate_fits(training, targets, validation, num_labels):
    for index, candidate in enumerate(CANDIDATES):
        classifier = fit_classifier(training, targets, candidate, num_labels)
        predictions = predict(classifier, validation)
        yield index, classifier, predictions


def selection_record(fold_index, method, chosen, scores):
    regularization, weight = CANDIDATES[chosen]
    return {'fold': fold_index, 'method': method, 'C': regularization,
            'class_weight': weight, 'pooled_validation_macro_f1': scores[chosen],
            'candidate_scores': scores}


def best_candidate(truth, candidate_predictions, labels):
    scores = []
    best_score = -1.0
    chosen = 0
    for index, predictions in enumerate(candidate_predictions):
        score = macro_f1(truth, predictions, labels)
        scores.append(score)
        if score > best_score + 1e-12:
            best_score = score
            chosen = index
    return chosen, scores


def append_predictions(result, method, ids, predictions, by_id, source, fold_index):
    for sid, predicted in zip(ids, predictions):
        record = {'sample_id': sid, 'reference_label_id': by_id[sid]['label_id'],
                  'predicted_label_id': int(predicted), 'group_id': source[sid],
                  'fold': fold_index}
        result['predictions'][method].append(record)


def train_holdout(fold, context, fraction, subset_seed, fold_index, result):
    by_id, positions, embeddings, plan = context
    k = plan['num_labels']
    train_ids = select_training_ids(fold['train'], by_id, fraction, subset_seed, k)
    validation_ids = fold['validation']
    vectorizer, matrices = fit_features(train_ids, validation_ids, by_id, positions, embeddings)
    training_targets = targets_for(train_ids, by_id)
    validation_targets = targets_for(validation_ids, by_id)
    labels = fold['validation_labels']
    selected_models = {}
    for method in METHODS:
        training, validation = matrices[method]
        best_score = -1.0
        chosen = 0
        scores = []
        for index, classifier, predictions in candidate_fits(training, training_targets, validation, k):
            score = macro_f1(validation_targets, predictions, labels)
            scores.append(score)
            if score > best_score + 1e-12:
                best_score = score
                chosen = index
                selected_models[method] = classifier
        record = selection_record(fold_index, method, chosen, scores)
        result['selections'].append(record)
    test = query_features(vectorizer, fold['test'], by_id, positions, embeddings)
    source = plan['sample_to_source_group']
    for method in METHODS:
        predictions = predict(selected_models[method], test[method])
        append_predictions(result, method, fold['test'], predictions, by_id, source, fold_index)
    result['training_subsets'].append({'fold': fold_index, 'requested_fraction': fraction,
        'selected_rows': len(train_ids), 'full_training_rows': len(fold['train']),
        'selected_sample_ids': train_ids, 'subset_seed': subset_seed})


def inner_predictions(fold, context):
    by_id, positions, embeddings, plan = context
    pooled_truth = []
    pooled_predictions = {}
    for method in METHODS:
        pooled_predictions[method] = []
        for candidate in CANDIDATES:
            pooled_predictions[method].append([])
    for inner in fold['inner']:
        train_ids = inner['train']
        validation_ids = inner['validation']
        _, matrices = fit_features(train_ids, validation_ids, by_id, positions, embeddings)
        training_targets = targets_for(train_ids, by_id)
        validation_targets = targets_for(validation_ids, by_id)
        pooled_truth.extend(validation_targets.tolist())
        for method in METHODS:
            training, validation = matrices[method]
            for index, classifier, predictions in candidate_fits(training, training_targets, validation, plan['num_labels']):
                pooled_predictions[method][index].extend(predictions.tolist())
    return pooled_truth, pooled_predictions


def train_nested(fold, context, fold_index, result):
    by_id, positions, embeddings, plan = context
    truth, predictions = inner_predictions(fold, context)
    chosen_by_method = {}
    for method in METHODS:
        chosen, scores = best_candidate(truth, predictions[method], plan['evaluation_labels'])
        chosen_by_method[method] = chosen
        result['selections'].append(selection_record(fold_index, method, chosen, scores))
    _, matrices = fit_features(fold['train'], fold['test'], by_id, positions, embeddings)
    training_targets = targets_for(fold['train'], by_id)
    source = plan['sample_to_source_group']
    for method in METHODS:
        training, test = matrices[method]
        candidate = CANDIDATES[chosen_by_method[method]]
        classifier = fit_classifier(training, training_targets, candidate, plan['num_labels'])
        outer_predictions = predict(classifier, test)
        append_predictions(result, method, fold['test'], outer_predictions, by_id, source, fold_index)
    result['training_subsets'].append({'fold': fold_index, 'requested_fraction': 1.0,
        'selected_rows': len(fold['train']), 'full_training_rows': len(fold['train'])})


def train_experiment(rows, plan, embedding_path, fraction=1.0, subset_seed=17):
    by_id = validate_plan(plan, rows)
    positions, embeddings = load_embeddings(embedding_path, rows)
    if plan['mode'] == 'nested' and fraction != 1.0:
        raise ValueError('The nested protocol uses complete outer training pools')
    result = {'mode': plan['mode'], 'num_labels': plan['num_labels'],
              'evaluation_labels': plan['evaluation_labels'], 'test_labels': {},
              'predictions': {}, 'selections': [], 'training_subsets': []}
    for method in METHODS:
        result['predictions'][method] = []
    context = (by_id, positions, embeddings, plan)
    for index, fold in enumerate(plan['folds']):
        labels = plan['evaluation_labels']
        if 'test_labels' in fold:
            labels = fold['test_labels']
        result['test_labels'][str(index)] = labels
        if plan['mode'] == 'nested':
            train_nested(fold, context, index, result)
        else:
            train_holdout(fold, context, fraction, subset_seed, index, result)
        print('Completed fold', index, 'for', plan['mode'], flush=True)
    return result
