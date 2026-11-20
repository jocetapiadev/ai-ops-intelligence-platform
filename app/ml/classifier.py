from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.ml.preprocessing import clean_text

TRAINING_DATA: list[tuple[str, str]] = [
    ("postgres connection pool exhausted timeout sql query slow", "database"),
    ("database deadlock replication lag disk full db", "database"),
    ("pod crashloopbackoff oomkilled kubernetes node not ready", "kubernetes"),
    ("deployment rollout failed ingress controller liveness probe", "kubernetes"),
    ("high latency packet loss dns timeout firewall packet drop", "network"),
    ("load balancer 502 bad gateway tls handshake timeout", "network"),
    ("jwt token expired unauthorized 401 oauth permission denied", "auth"),
    ("sso login failed invalid credentials role access denied", "auth"),
]


class IncidentClassifier:
    def __init__(self) -> None:
        self.pipeline: Pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        )
        self._is_fitted = False

    def fit(self, samples: list[tuple[str, str]] | None = None) -> None:
        dataset = samples or TRAINING_DATA
        texts = [clean_text(text) for text, _ in dataset]
        labels = [label for _, label in dataset]
        self.pipeline.fit(texts, labels)
        self._is_fitted = True

    def predict(self, text: str) -> tuple[str, float]:
        if not self._is_fitted:
            self.fit()
        cleaned = clean_text(text)
        prediction = self.pipeline.predict([cleaned])[0]
        probabilities = self.pipeline.predict_proba([cleaned])[0]
        confidence = float(max(probabilities))
        return str(prediction), round(confidence, 4)


classifier = IncidentClassifier()
