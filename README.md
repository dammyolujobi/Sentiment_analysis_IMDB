# IMDB Sentiment Analysis with NLP and Scikit-learn

This project is a machine learning pipeline for sentiment analysis on the [IMDB Movie Reviews Dataset](https://ai.stanford.edu/~amaas/data/sentiment/). It preprocesses raw text, extracts various features (TF-IDF, text length, lexicon scores), and uses a grid-searched classifier to predict sentiment. Additionally, a soft-voting ensemble model is implemented to enhance predictive performance.

---

## 📂 Dataset

- `IMDB-Dataset.csv`  
  A dataset of 50,000 labeled movie reviews (`positive` or `negative`).

---

## 🧹 Preprocessing Steps

1. **Lowercasing**
2. **Contraction Expansion**
3. **HTML Tag Removal**
4. **Non-alphabetic Character Removal**
5. **Negation Handling**
6. **Tokenization, Stopword Removal, and Lemmatization**

---

## 🔍 Features Used

- **TF-IDF Vectorization** (with configurable `ngram_range`, `min_df`, `max_df`)
- **Text Length**: Total word count in each review
- **Lexicon Score**: Sum of sentiment values from a predefined lexicon (AFINN-like)

---

## 🤖 Models

- **Baseline**: `LogisticRegression` with class balancing and grid search
- **GridSearchCV** for hyperparameter tuning
- **Soft-Voting Ensemble**:
  - Logistic Regression (best model)
  - Multinomial Naive Bayes
  - Calibrated Linear SVM

---

## 🧪 Evaluation Metrics

- Accuracy
- F1 Score (used in cross-validation scoring)
- ROC AUC
- Confusion Matrix
- Misclassified example analysis

---

## 📈 Results

- Best model from grid search: **Logistic Regression**
- Ensemble voting classifier shows improved accuracy

---

## 🚀 How to Run

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Ensure you have the `IMDB-Dataset.csv` in the root directory.

3. Run the pipeline:
    ```bash
    python sentiment_analysis.py
    ```

---

## 🔧 TODO

- Expand and integrate more comprehensive sentiment lexicons
- Support for other datasets
- Deploy with FastAPI or Flask for inference

---

## 🛠 Requirements

- pandas
- numpy
- scikit-learn
- nltk
- contractions

To download NLTK resources:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
