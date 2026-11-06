# Comment Category Prediction

Machine learning project for predicting the category assigned to online comments using **NLP, numerical features, categorical metadata, and engagement signals**.

This project was developed as part of the **IIT Madras BS in Data Science and Applications MLP project** and the **Kaggle Comment Category Prediction Challenge**.

---

## 📌 Project Overview

The objective is to predict the final category (`label`) assigned to each comment.

The dataset combines:

- 💬 Comment text
- 📅 Timestamp information
- 🧵 Discussion/post identifiers
- 👍 Upvotes and 👎 downvotes
- 😀 Emoticon indicators
- 🔎 Internal platform features
- 🏷️ Categorical metadata related to detected topics

The task is a **four-class classification problem**.

The project explores different machine learning approaches and compares their performance using **macro F1-score**.

---

## 📊 Dataset

The competition provides three files:

| File | Description |
|---|---|
| `train.csv` | Training data containing features and the target `label` |
| `test.csv` | Test data containing features without the target |
| `sample_submission.csv` | Example submission format |

### Dataset Size

- **Training samples:** 198,000
- **Test samples:** 102,000
- **Training features:** 14 input features + target
- **Target classes:** 4

### Main Features

| Feature Group | Features |
|---|---|
| Text | `comment` |
| Temporal | `created_date` |
| Discussion | `post_id` |
| Engagement | `upvote`, `downvote` |
| Emoticons | `emoticon_1`, `emoticon_2`, `emoticon_3` |
| Internal features | `if_1`, `if_2` |
| Categorical metadata | `race`, `religion`, `gender` |
| Boolean | `disability` |
| Target | `label` |

The dataset contains missing values in several categorical columns, which were handled during preprocessing.

---

## 🔍 Exploratory Data Analysis

The notebook performs exploratory analysis to understand:

- Dataset structure and feature types
- Missing values
- Duplicate records
- Numerical feature distributions
- Target class distribution
- Categorical feature distributions
- Comment length and text characteristics
- Relationships between features and the target
- Correlations between numerical features

No duplicate rows were found in either the training or test dataset.

---

## 🛠️ Feature Engineering & Preprocessing

### Text Features

The `comment` column was transformed using **TF-IDF** with both word-level and character-level features.

#### Word-level TF-IDF

- N-gram range: `(1, 2)`
- Maximum features: `23,000`
- Minimum document frequency: `3`
- Maximum document frequency: `0.9`
- English stop-word removal
- Sublinear TF scaling

#### Character-level TF-IDF

- Analyzer: `char_wb`
- N-gram range: `(3, 5)`
- Maximum features: `15,000`

The word and character TF-IDF representations were combined using `FeatureUnion`.

### Numerical Features

Numerical features were standardized using `StandardScaler`.

For the Naive Bayes pipeline, `MinMaxScaler` was used instead.

### Categorical Features

The following categorical features were handled using one-hot encoding:

```text
race
religion
gender
```

Missing categorical values were replaced with `"unknown"`.

The boolean `disability` feature was converted to an integer representation.

---

## 🤖 Models

Three classification models were evaluated during the project.

### 1. Linear SVM

```text
LinearSVC
C = 0.5
max_iter = 800
```

**Validation Macro F1:** `0.6875`

---

### 2. Logistic Regression

```text
LogisticRegression
C = 1
class_weight = "balanced"
max_iter = 400
```

**Validation Macro F1:** `0.7990`

This was the strongest of the three evaluated baseline models.

---

### 3. Multinomial Naive Bayes

```text
MultinomialNB
```

**Validation Macro F1:** `0.5869`

---

## 📈 Model Comparison

| Model | Macro F1 |
|---|---:|
| Linear SVM | 0.6875 |
| Logistic Regression | **0.7990** |
| Multinomial Naive Bayes | 0.5869 |

The experiments showed that **Logistic Regression performed best among the evaluated models** on the validation split.

---

## 🧪 Experimentation

The repository separates experimentation from the final notebook.

### Experiments

`notebooks/comment_category_prediction_experiments.ipynb`

Contains exploratory work, model trials, preprocessing experiments, and approaches that were evaluated during development.

### Final Notebook

`notebooks/comment_category_prediction_final.ipynb`

Contains the cleaned-up version of the selected approach and the final prediction/submission workflow.

---

## 📁 Repository Structure

```text
comment-category-prediction/
│
├── notebooks/
│   ├── comment_category_prediction_experiments.ipynb
│   └── comment_category_prediction_final.ipynb
│
├── README.md
└── .gitignore
```

---

## 🧰 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- TF-IDF
- Linear SVM
- Logistic Regression
- Multinomial Naive Bayes
- Kaggle

---

## 💡 Key Takeaways

- Combined **textual and structured features** for classification.
- Used both **word-level and character-level TF-IDF** representations.
- Compared multiple classical machine learning models.
- Used **macro F1-score** to evaluate multi-class classification performance.
- Logistic Regression achieved the strongest validation performance among the tested models with a macro F1-score of **0.7990**.
- The project provided practical experience with building an end-to-end NLP classification pipeline on a large dataset.

---

## 🏆 Competition

**Comment Category Prediction Challenge**

Hosted on Kaggle by `iitmbscs2008p`.

The competition focused on predicting the category assigned to user-generated comments based on textual content, metadata, interaction signals, and internal features.

---

## 👤 Author

**Pallavi**

IIT Madras BS in Data Science and Applications

---
