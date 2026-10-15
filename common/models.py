"""Fixed lexical, semantic, and combined text classifiers."""
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

METHODS = ("lexical", "semantic", "combined")
CANDIDATES = (
    (0.25, None),
    (0.25, "balanced"),
    (1.0, None),
    (1.0, "balanced"),
    (4.0, None),
    (4.0, "balanced"),
)


def make_features():
    word_features = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=40000,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True,
        min_df=1,
        lowercase=True,
        dtype=np.float32,
    )
    character_features = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=60000,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True,
        min_df=1,
        lowercase=True,
        dtype=np.float32,
    )
    return FeatureUnion([
        ("word", word_features),
        ("char", character_features),
    ])


def combine(method, lexical, embedding=None):
    if method == "lexical":
        return lexical
    if method not in METHODS:
        raise ValueError("Unknown method")

    semantic_features = np.asarray(embedding, dtype=np.float32)
    if semantic_features.ndim != 2:
        raise ValueError("Embeddings must be a matrix")
    if semantic_features.shape[0] != lexical.shape[0]:
        raise ValueError("Lexical and semantic row counts differ")
    if not np.isfinite(semantic_features).all():
        raise ValueError("Embeddings contain nonfinite values")
    row_norms = np.linalg.norm(semantic_features, axis=1)
    if not np.allclose(row_norms, 1.0, atol=2e-4):
        raise ValueError("Embeddings must have unit length")

    scale = np.float32(np.sqrt(2.0))
    if method == "semantic":
        return semantic_features * scale
    lexical_features = lexical / scale
    semantic_features = sparse.csr_matrix(semantic_features)
    return sparse.hstack(
        [lexical_features, semantic_features],
        format="csr",
        dtype=np.float32,
    )


def fit_classifier(X, y, candidate, num_labels):
    targets = np.asarray(y)
    expected_labels = set(range(num_labels))
    if set(targets.tolist()) != expected_labels:
        raise ValueError("Every output label needs a training example")

    regularization, class_weight = candidate
    classifier = LinearSVC(
        C=regularization,
        class_weight=class_weight,
        random_state=0,
        penalty="l2",
        loss="squared_hinge",
        dual=True,
        max_iter=20000,
        tol=1e-4,
    )
    classifier.fit(X, targets)
    if set(classifier.classes_.tolist()) != expected_labels:
        raise ValueError("Classifier labels are incomplete")
    return classifier


def aligned_scores(classifier, X):
    num_labels = len(classifier.classes_)
    expected_labels = set(range(num_labels))
    if set(classifier.classes_.tolist()) != expected_labels:
        raise ValueError("Classifier labels must be consecutive integers")

    scores = classifier.decision_function(X)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    aligned = np.empty((X.shape[0], num_labels), dtype=np.float32)
    aligned[:, classifier.classes_] = scores
    if not np.isfinite(aligned).all():
        raise ValueError("Classifier scores contain nonfinite values")
    return aligned


def predict(classifier, X):
    scores = aligned_scores(classifier, X)
    return scores.argmax(axis=1)


def macro_f1(y, pred, eval_labels):
    if len(eval_labels) == 0:
        raise ValueError("No evaluation labels")
    score = f1_score(
        y,
        pred,
        labels=list(eval_labels),
        average="macro",
        zero_division=0,
    )
    return float(score)
