from collections import defaultdict
from common.allocation import allocate_splits, fallback_milp
from common.io import index_rows, label_ids, sample_ids

DEFAULT_SEEDS = (10517, 10529, 10543)
PARTITIONS = ('train', 'validation', 'test')


def validate_vocabulary(rows):
    index_rows(rows)
    projects = set()
    names_by_id = {}
    ids_by_name = {}
    for row in rows:
        projects.add(row['project_id'])
        label_id = row['label_id']
        name = row['label']
        if label_id in names_by_id and names_by_id[label_id] != name:
            raise ValueError('One label ID has several names')
        if name in ids_by_name and ids_by_name[name] != label_id:
            raise ValueError('One label name has several IDs')
        names_by_id[label_id] = name
        ids_by_name[name] = label_id
    if len(projects) != 1:
        raise ValueError('Run each project separately')
    count = len(names_by_id)
    if set(names_by_id) != set(range(count)):
        raise ValueError('Label IDs must be consecutive from zero')
    return count


def text_groups(rows):
    mapping = {}
    for row in rows:
        mapping[row['sample_id']] = row['leakage_group_id']
    return mapping


def new_plan(rows, mode, groups, evaluation_labels):
    return {'mode': mode, 'num_labels': validate_vocabulary(rows),
            'pool_ids': sample_ids(rows), 'evaluation_labels': evaluation_labels,
            'sample_to_source_group': groups, 'partition_groups': groups,
            'fixed_groups': [], 'folds': []}


def validate_parts(parts, required_ids, by_id, groups, num_labels):
    seen_ids = set()
    seen_groups = set()
    for ids in parts:
        current_ids = set(ids)
        if not ids or len(current_ids) != len(ids):
            raise ValueError('Each partition needs unique, nonempty sample IDs')
        if not current_ids <= required_ids:
            raise ValueError('A partition contains an unknown sample ID')
        if seen_ids & current_ids:
            raise ValueError('Samples cross partitions')
        current_groups = set()
        for sid in ids:
            current_groups.add(groups[sid])
        if seen_groups & current_groups:
            raise ValueError('Groups cross partitions')
        seen_ids.update(current_ids)
        seen_groups.update(current_groups)
    if seen_ids != required_ids:
        raise ValueError('Partitions do not cover their input pool')
    training_labels = set()
    for sid in parts[0]:
        training_labels.add(by_id[sid]['label_id'])
    if training_labels != set(range(num_labels)):
        raise ValueError('Every fitting partition must retain all output labels')


def initial_plan(train_rows, validation_rows, test_rows):
    rows = train_rows + validation_rows + test_rows
    groups = text_groups(rows)
    test_labels = label_ids(test_rows)
    plan = new_plan(rows, 'initial', groups, test_labels)
    fold = {'train': sample_ids(train_rows), 'validation': sample_ids(validation_rows),
            'test': sample_ids(test_rows), 'validation_labels': label_ids(validation_rows),
            'test_labels': test_labels}
    plan['folds'].append(fold)
    validate_plan(plan, rows)
    return plan, rows


def support_by_label(rows, groups):
    support = defaultdict(set)
    for row in rows:
        support[row['label_id']].add(groups[row['sample_id']])
    return support


def document_label_scope(rows, groups):
    support = support_by_label(rows, groups)
    train_only = set()
    for label, positive_groups in support.items():
        if len(positive_groups) <= 2:
            train_only.add(label)
    while True:
        fixed_groups = set()
        for label in train_only:
            fixed_groups.update(support[label])
        additional = set()
        for label, positive_groups in support.items():
            if label in train_only:
                continue
            available = positive_groups - fixed_groups
            if len(available) < 2:
                additional.add(label)
        if not additional:
            evaluation = set(support) - train_only
            return sorted(evaluation), fixed_groups
        train_only.update(additional)


def allocator_inputs(rows, groups):
    samples = []
    for row in rows:
        samples.append({'sample_id': row['sample_id'], 'project_id': row['project_id'],
                        'labels_observed': [row['label']], 'core_eligible': True,
                        'leakage_group_id': groups[row['sample_id']]})
    group_records = []
    for group in sorted(set(groups.values())):
        group_records.append({'leakage_group_id': group})
    return samples, group_records


def holdout_folds(rows, groups, evaluation, forced, seeds, allow_fallback):
    names = set()
    for row in rows:
        if row['label_id'] in evaluation:
            names.add(row['label'])
    folds = []
    for seed in seeds:
        samples, group_records = allocator_inputs(rows, groups)
        try:
            audit = allocate_splits(samples, group_records, forced, names, seed)
        except ValueError:
            if not allow_fallback:
                raise
            audit = fallback_milp(samples, forced, names, seed)
        fold = {'seed': seed, 'allocation': audit}
        for partition in PARTITIONS:
            ids = []
            for sample in samples:
                if sample['split'] == partition:
                    ids.append(sample['sample_id'])
            fold[partition] = ids
        fold['validation_labels'] = list(evaluation)
        fold['test_labels'] = list(evaluation)
        folds.append(fold)
    return folds


def text_resplit_plan(train_rows, validation_rows, test_rows, seeds=DEFAULT_SEEDS):
    initial_plan(train_rows, validation_rows, test_rows)
    rows = train_rows + validation_rows
    groups = text_groups(rows)
    support = support_by_label(rows, groups)
    evaluation = []
    for label, positive_groups in support.items():
        if len(positive_groups) >= 3:
            evaluation.append(label)
    evaluation.sort()
    plan = new_plan(rows, 'text_resplit', groups, evaluation)
    plan['folds'] = holdout_folds(rows, groups, evaluation, set(), seeds, False)
    validate_plan(plan, rows)
    return plan, rows


def selected_ids(rows, source_groups, chosen_groups):
    ordered = sorted(rows, key=sample_order)
    ids = []
    for row in ordered:
        if source_groups[row['sample_id']] in chosen_groups:
            ids.append(row['sample_id'])
    return ids


def sample_order(row):
    return row['sample_id']


def groups_in_fold(assignment, fold):
    groups = set()
    for group, assigned_fold in assignment.items():
        if assigned_fold == fold:
            groups.add(group)
    return groups


def nested_folds(rows, source, fixed, witness):
    eligible = set(source.values()) - fixed
    if witness is None:
        raise ValueError('Complete coverage requires a recorded group witness')
    assignments = witness['outer_fold_by_source_group']
    if set(assignments) != eligible or set(assignments.values()) != {0, 1, 2}:
        raise ValueError('Outer witness must partition all eligible groups into three folds')
    if set(witness['fixed_training_source_groups']) != fixed:
        raise ValueError('Witness training-only groups differ from label-support closure')
    folds = []
    for outer in range(3):
        test_groups = groups_in_fold(assignments, outer)
        remaining = eligible - test_groups
        inner_assignment = witness['inner_validation_fold_by_source_group'][str(outer)]
        if set(inner_assignment) != remaining or set(inner_assignment.values()) != {0, 1}:
            raise ValueError('Inner witness must partition eligible outer training groups')
        inner_folds = []
        for inner in range(2):
            validation_groups = groups_in_fold(inner_assignment, inner)
            training_groups = remaining - validation_groups
            training_groups.update(fixed)
            inner_folds.append({'train': selected_ids(rows, source, training_groups),
                                'validation': selected_ids(rows, source, validation_groups)})
        outer_training = remaining | fixed
        folds.append({'train': selected_ids(rows, source, outer_training),
                      'test': selected_ids(rows, source, test_groups), 'inner': inner_folds})
    return folds


def document_plan(rows, mapping, mode='source', witness=None, seeds=DEFAULT_SEEDS):
    if mode not in ('source', 'text', 'nested'):
        raise ValueError('Unknown document experiment mode')
    source = {}
    for row in rows:
        sid = row['sample_id']
        source[sid] = mapping['sample_to_source_group'][sid]
    evaluation, fixed = document_label_scope(rows, source)
    plan = new_plan(rows, mode, source, evaluation)
    plan['fixed_groups'] = sorted(fixed)
    if mode == 'nested':
        plan['folds'] = nested_folds(rows, source, fixed, witness)
    else:
        partition_groups = source
        if mode == 'text':
            partition_groups = text_groups(rows)
        forced = set()
        for row in rows:
            if row['label_id'] not in evaluation:
                forced.add(partition_groups[row['sample_id']])
        plan['partition_groups'] = partition_groups
        plan['folds'] = holdout_folds(rows, partition_groups, evaluation, forced, seeds, True)
    validate_plan(plan, rows)
    return plan


def validate_plan(plan, rows):
    by_id = index_rows(rows)
    num_labels = validate_vocabulary(rows)
    ids = set(by_id)
    if set(plan['pool_ids']) != ids or len(plan['pool_ids']) != len(ids):
        raise ValueError('Plan sample IDs do not match the prepared pool')
    if plan['num_labels'] != num_labels:
        raise ValueError('Plan output-label count differs from the prepared pool')
    groups = plan['partition_groups']
    source = plan['sample_to_source_group']
    if set(groups) != ids or set(source) != ids:
        raise ValueError('Incomplete grouping map')
    evaluation = set(plan['evaluation_labels'])
    if not evaluation or not evaluation <= set(range(num_labels)):
        raise ValueError('Invalid evaluation-label scope')
    if not plan['folds']:
        raise ValueError('A plan needs at least one fold')
    if plan['mode'] == 'nested':
        validate_nested_plan(plan, by_id, groups, source, ids, num_labels)
    else:
        for fold in plan['folds']:
            parts = [fold['train'], fold['validation'], fold['test']]
            validate_parts(parts, ids, by_id, groups, num_labels)
            for partition in ('validation', 'test'):
                labels = set()
                for sid in fold[partition]:
                    labels.add(by_id[sid]['label_id'])
                if labels != set(fold[partition + '_labels']):
                    raise ValueError('Declared label scope differs from holdout support')
    return by_id


def validate_nested_plan(plan, by_id, groups, source, ids, num_labels):
    if groups != source or len(plan['folds']) != 3:
        raise ValueError('Nested evaluation requires three document-grouped outer folds')
    fixed = set(plan['fixed_groups'])
    eligible_ids = set()
    for sid in ids:
        if source[sid] not in fixed:
            eligible_ids.add(sid)
    tested = []
    for fold in plan['folds']:
        validate_parts([fold['train'], fold['test']], ids, by_id, groups, num_labels)
        if len(fold['inner']) != 2:
            raise ValueError('Nested selection requires two inner validation folds')
        tested.extend(fold['test'])
        validated = []
        for inner in fold['inner']:
            parts = [inner['train'], inner['validation']]
            validate_parts(parts, set(fold['train']), by_id, groups, num_labels)
            validated.extend(inner['validation'])
        expected = set(fold['train']) & eligible_ids
        if len(validated) != len(set(validated)) or set(validated) != expected:
            raise ValueError('Inner validation must cover non-pinned outer training rows once')
    if len(tested) != len(set(tested)) or set(tested) != eligible_ids:
        raise ValueError('Outer tests must cover each non-pinned row exactly once')
