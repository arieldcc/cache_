import numpy as np, pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM=True
except Exception:
    _HAS_LGBM=False

class ELMClassifier:
    def __init__(self, n_hidden=256, activation='relu', reg_lambda=1e-2, random_state=123):
        # Coerce types so YAML strings like "1e-2" don't break NumPy
        self.n_hidden = int(n_hidden)
        self.activation = str(activation)
        self.reg_lambda = float(reg_lambda)
        self.random_state = np.random.RandomState(int(random_state))
        self.W=None; self.b=None; self.beta=None
    def _act(self, X):
        Z = X @ self.W + self.b
        if self.activation=='relu': return np.maximum(0.0, Z)
        if self.activation=='tanh': return np.tanh(Z)
        return 1.0/(1.0+np.exp(-Z))
    def fit(self, X, y):
        X=np.asarray(X, dtype=float); y=np.asarray(y, dtype=float).reshape(-1,1)
        n_features=X.shape[1]
        self.W=self.random_state.normal(scale=1.0/np.sqrt(n_features), size=(n_features, self.n_hidden))
        self.b=self.random_state.normal(scale=0.1, size=(1,self.n_hidden))
        H=self._act(X)
        lamI=self.reg_lambda*np.eye(self.n_hidden)
        self.beta=np.linalg.pinv(H.T@H+lamI)@H.T@y
        return self
    def predict_proba(self, X):
        H=self._act(np.asarray(X, dtype=float)); s=H@self.beta; p=1.0/(1.0+np.exp(-s))
        p=np.clip(p,1e-6,1-1e-6); return np.hstack([1-p,p])

def make_lr_sgd_with_decay(eta0=0.05, power_t=0.5, max_iter=20, alpha=1e-4, class_weight='balanced', random_state=123):
    clf=SGDClassifier(loss='log_loss', learning_rate='invscaling', eta0=float(eta0), power_t=float(power_t),
                      alpha=float(alpha), max_iter=int(max_iter), tol=1e-3,
                      class_weight=class_weight, random_state=int(random_state))
    return Pipeline([('scaler', StandardScaler(with_mean=False)), ('sgd', clf)])

def make_gbdt(params, random_state=123):
    if _HAS_LGBM: return LGBMClassifier(random_state=random_state, **params)
    return HistGradientBoostingClassifier(random_state=random_state)

def make_nb(nb_type='gaussian'):
    return BernoulliNB() if str(nb_type)=='bernoulli' else GaussianNB()

class SoftVotingEnsemble:
    def __init__(self, models, weights): self.models=models; self.weights=weights
    def fit(self, X, y):
        for m in self.models.values(): m.fit(X,y)
        return self
    def predict_proba(self, X):
        total_w=0.0; agg=None
        for name,m in self.models.items():
            w=float(self.weights.get(name,1.0)); p=m.predict_proba(X)[:,1]
            agg = w*p if agg is None else agg + w*p; total_w += w
        p1 = agg / max(total_w,1e-6); p1 = np.clip(p1,1e-6,1-1e-6)
        return np.vstack([1-p1,p1]).T
