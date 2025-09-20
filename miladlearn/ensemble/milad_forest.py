import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier


class MiladClassifier(BaseEstimator, ClassifierMixin):
    """
    MiladClassifier
    ----------------
    نسخه‌ی سفارشی از RandomForest که علاوه بر فیچرهای اصلی،
    نسبت فیچرها (feature ratios) رو هم به ورودی اضافه می‌کنه.
    """

    def __init__(self, n_estimators=100, random_state=None):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None

    def _add_ratios(self, X):
        X = np.array(X)
        n_features = X.shape[1]
        ratios = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                ratio = X[:, i] / (X[:, j] + 1e-9)  # جلوگیری از تقسیم بر صفر
                ratios.append(ratio.reshape(-1, 1))
        return np.hstack([X] + ratios)

    def fit(self, X, y):
        X_new = self._add_ratios(X)
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )
        self.model.fit(X_new, y)
        return self

    def predict(self, X):
        X_new = self._add_ratios(X)
        return self.model.predict(X_new)

    def predict_proba(self, X):
        X_new = self._add_ratios(X)
        return self.model.predict_proba(X_new)
