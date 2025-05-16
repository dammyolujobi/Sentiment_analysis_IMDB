import re
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import FunctionTransformer
import contractions
from sklearn.calibration import CalibratedClassifierCV


# --- Preprocessing utilities ---
stopwords_english = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

negation_terms = {"not", "no", "never", "n't"}
def expand_contractions(text):
    return contractions.fix(text)

def handle_negation(text):
    tokens = word_tokenize(text)
    output = []
    negate = False
    for t in tokens:
        if t in negation_terms:
            negate = True
            output.append(t)
        elif negate:
            output.append(f"{t}_NEG")
            negate = False
        else:
            output.append(t)
    return ' '.join(output)

def preprocess_text(text):
    text = text.lower()
    text = expand_contractions(text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = handle_negation(text)
    tokens = word_tokenize(text)
    filtered = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in stopwords_english]
    return ' '.join(filtered)

# --- Feature transformers ---
class TextLength(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        lengths = X.apply(lambda text: len(text.split()))
        return lengths.to_frame(name='length')

class LexiconScore(BaseEstimator, TransformerMixin):
    def __init__(self, lexicon):
        self.lexicon = lexicon
    def fit(self, X, y=None): return self
    def transform(self, X):
        scores = X.apply(lambda text: sum(self.lexicon.get(w, 0) for w in text.split()))
        return scores.to_frame(name='lex_score')

# Load data
df = pd.read_csv('IMDB-Dataset.csv')
df['cleaned'] = df['review'].apply(preprocess_text)
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
X = df['cleaned']
y = df['label']

# Split
test_size = 0.2
random_state = 42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, stratify=y, random_state=random_state)

# Example lexicon (AFINN-like stub)
example_lex = {'good': 2, 'great': 3, 'bad': -2, 'terrible': -3}

# Pipeline
pipeline = Pipeline([
    ('features', FeatureUnion([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=5, max_df=0.8, max_features=10000, sublinear_tf=True)),
        ('length', Pipeline([
            ('extract', FunctionTransformer(lambda x: x, validate=False)),
            ('len_feat', TextLength())
        ])),
        ('lex', Pipeline([
            ('extract', FunctionTransformer(lambda x: x, validate=False)),
            ('lex_feat', LexiconScore(example_lex))
        ])),
    ])),
    ('clf', LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=1000))
])

# Grid search params
grid_params = {
    'features__tfidf__ngram_range': [(1,1), (1,2)],
    'features__tfidf__min_df': [3,5],
    'clf__C': [0.1, 1, 10]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
grid = GridSearchCV(pipeline, grid_params, cv=cv, scoring='f1', n_jobs=-1, verbose=2)

grid.fit(X_train, y_train)
print("Best parameters:", grid.best_params_)

# Evaluate
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC:\n", roc_auc_score(y_test, best_model.predict_proba(X_test)[:,1]))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Error Analysis
mis = df.loc[X_test.index][y_test != y_pred]
print("\nSome misclassified examples:\n")
for idx, row in mis.sample(5, random_state=random_state).iterrows():
    print(f"Review: {row['review'][:200]}...")
    pred_label = best_model.predict(pd.Series([row['cleaned']]))[0]
    pred_sentiment = 'positive' if pred_label == 1 else 'negative'
    print(f"True: {row['sentiment']}, Pred: {pred_sentiment}\n")


# Optional: ensemble classifier
ensemble = VotingClassifier([
    ('lr', grid.best_estimator_),
    ('nb', Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=5, max_df=0.8, sublinear_tf=True)),
        ('clf', MultinomialNB())
    ])),
    ('svm', Pipeline([
    ('vect', TfidfVectorizer(ngram_range=(1,2), max_features=5000)),
    ('clf', CalibratedClassifierCV(LinearSVC(), cv=3))
])),
], voting='soft', weights=[2,1,1], n_jobs=-1)
ensemble.fit(X_train, y_train)
pos = ensemble.predict(X_test)
print(f"Ensemble Accuracy: {accuracy_score(y_test, pos):.4f}")