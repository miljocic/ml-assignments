# Machine Learning Assignments


A collection of machine learning assignments implemented from scratch in Python using NumPy and TensorFlow. Covers polynomial regression, regularization, K-Nearest Neighbors, Naive Bayes text classification, and ensemble/neural network methods.

---

## Tech Stack

- **Python 3** — core language
- **TensorFlow / Keras** — gradient-based optimization, neural networks
- **NumPy / Pandas** — data manipulation
- **scikit-learn** — train/test splits, label encoding, CountVectorizer, RandomForest, cross-validation
- **NLTK** — text preprocessing (stopwords, lemmatization)
- **Matplotlib / Seaborn** — visualizations

---

## Assignments

### Polynomial Regression (`bottle.csv` — ocean salinity vs temperature)

| Notebook | Description |
|----------|-------------|
| `2a.ipynb` | Polynomial regression degrees 1-6, trained via Adam optimizer. Plots cost vs degree. Degree 3 identified as optimal before diminishing returns. |
| `2b.ipynb` | L2 regularization on degree-4 polynomial. Tests lambda values [0, 0.001, 0.01, 0.1, 1, 10, 100]. Analyzes underfitting vs overfitting tradeoff. |

### Titanic Survival Classification (`train.csv`, `test.csv`)

| Notebook | Description |
|----------|-------------|
| `2_domaci_1_zadatak.ipynb` | Full preprocessing pipeline (outlier removal, IQR, age imputation by sex/class, feature engineering: Family, AgeBand, FareBand, label encoding). Random Forest (70 trees, depth 7) with 5-fold cross-validation. Feature importance visualization. |
| `2_domaci_2_zadatak.ipynb` | Same preprocessing + StandardScaler normalization. Feedforward neural network (2 hidden layers, 128 neurons each, Adam, lr=0.001, 50 epochs, batch=64) built with TensorFlow. |

### K-Nearest Neighbors (`iris.csv` — species classification)

| Notebook | Description |
|----------|-------------|
| `3a.ipynb` | KNN implemented with TensorFlow tensors. Decision boundary visualization with custom colormap. Uses 2 features. |
| `3b.ipynb` | KNN accuracy vs k (1-15) using 2 features. Finds optimal k=7. Analyzes bias-variance tradeoff. |
| `3c.ipynb` | Same as 3b but using all 4 features. Demonstrates significant accuracy and stability improvement. |

### Naive Bayes Text Classification (`disaster-tweets.csv`)

| Notebook | Description |
|----------|-------------|
| `4.ipynb` | Multinomial Naive Bayes from scratch. Bag-of-words (5000 features). Text preprocessing: lowercasing, URL/mention/hashtag removal, stopwords, lemmatization. ~79.5% avg accuracy over 3 runs. LR metric analysis of discriminative words. |

---

## Key Results

| Task | Model | Result |
|------|-------|--------|
| Polynomial regression | Degree 3, Adam | Optimal cost/complexity tradeoff |
| Regularization | L2, lambda=0.001-0.01 | Best balance before underfitting |
| Titanic classification | Random Forest (70 trees) | 5-fold CV accuracy |
| Titanic classification | Neural Network (2x128) | Adam, 50 epochs |
| KNN (2 features) | k=7 | Best accuracy on Iris |
| KNN (4 features) | k=7 | Noticeably higher accuracy |
| Disaster tweet NLP | Multinomial NB | ~79.5% avg accuracy |

---

## Datasets

- `bottle.csv` — California Cooperative Oceanic Fisheries (salinity & temperature)
- `train.csv` / `test.csv` — Titanic survival dataset
- `iris.csv` — Iris flower species classification
- `disaster-tweets.csv` — Kaggle NLP disaster tweets dataset

---

## How to Run

```bash
pip install numpy pandas tensorflow scikit-learn nltk matplotlib seaborn
jupyter notebook
```
