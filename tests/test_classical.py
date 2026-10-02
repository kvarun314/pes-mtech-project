from agentic_sentiment.baselines.classical import (
    evaluate,
    train_decision_tree,
    train_naive_bayes,
    train_svm,
)

TEXTS = [
    "terrible product broke immediately", "awful waste of money",
    "okay nothing special", "average performance",
    "really happy with this purchase", "excellent quality love it",
] * 5
LABELS = [1, 1, 3, 3, 5, 5] * 5


def test_decision_tree_trains_and_evaluates():
    pipeline = train_decision_tree(TEXTS, LABELS)
    metrics = evaluate(pipeline, TEXTS, LABELS)
    assert metrics["accuracy"] > 0.5
    assert {"accuracy", "precision", "recall", "f1"} <= metrics.keys()


def test_svm_sigmoid_trains_and_evaluates():
    pipeline = train_svm(TEXTS, LABELS)
    metrics = evaluate(pipeline, TEXTS, LABELS)
    assert metrics["accuracy"] > 0.5


def test_naive_bayes_trains_and_evaluates():
    pipeline = train_naive_bayes(TEXTS, LABELS)
    metrics = evaluate(pipeline, TEXTS, LABELS)
    assert metrics["accuracy"] > 0.5
