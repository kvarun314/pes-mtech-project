"""Classical baselines matching Wang et al. Table 3 configs (CLAUDE.md):
Decision Tree + CountVectorizer, SVM (sigmoid, gamma=1.0) + TfidfVectorizer,
Multinomial NB (alpha=0.2) + CountVectorizer."""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def train_decision_tree(texts: list[str], labels: list[int]) -> Pipeline:
    pipeline = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("clf", DecisionTreeClassifier(random_state=42)),
    ])
    pipeline.fit(texts, labels)
    return pipeline


def train_svm(texts: list[str], labels: list[int]) -> Pipeline:
    pipeline = Pipeline([
        ("vectorizer", TfidfVectorizer()),
        ("clf", SVC(kernel="sigmoid", gamma=1.0)),
    ])
    pipeline.fit(texts, labels)
    return pipeline


def train_naive_bayes(texts: list[str], labels: list[int]) -> Pipeline:
    pipeline = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("clf", MultinomialNB(alpha=0.2)),
    ])
    pipeline.fit(texts, labels)
    return pipeline


def evaluate(pipeline: Pipeline, texts: list[str], labels: list[int]) -> dict:
    pred = pipeline.predict(texts)
    return {
        "accuracy": accuracy_score(labels, pred),
        "precision": precision_score(labels, pred, average="macro", zero_division=0),
        "recall": recall_score(labels, pred, average="macro", zero_division=0),
        "f1": f1_score(labels, pred, average="macro", zero_division=0),
    }
