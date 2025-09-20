import numpy as np
from miladlearn.ensemble import MiladClassifier


def test_milad_classifier():
    X = np.array([
        [1, 2, 3],
        [2, 4, 6],
        [3, 1, 2],
        [4, 2, 1]
    ])
    y = np.array([0, 1, 0, 1])

    clf = MiladClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    preds = clf.predict(X)

    assert len(preds) == len(y)
