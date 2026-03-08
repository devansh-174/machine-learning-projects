#2.Fake News Detection
# fake news is false information and hoaxes spread through social 
# media and other online media to achieve a political agenda. In this
# data science project idea, we will use Python to build a model 
# that can accurately detect whether a piece of news is real or fake.


# ===============================
# 0. importing the important libraries which are needed in this project
# ===============================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.dummy import DummyClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import PassiveAggressiveClassifier

# ===============================
# 1. Load Dataset
# ===============================
news_data = pd.read_csv('/Users/devanshbansal/Downloads/fake_news_dataset.csv')
print("Unique labels:", news_data["label"].unique())
print(news_data["label"].value_counts())

# ===============================
# 2. Data Cleaning & Preprocessing through EDA
# ===============================
# first we will check for null values in our dataset
print(news_data.isnull().sum())


#replacing null values with empty strings of author
news_data['author'] = news_data['author'].fillna('unknown')


#replacing null values with empty strings of source
news_data['source'] = news_data['source'].fillna('unknown')

#checking for null values again
print(news_data.isnull().sum())

#dropping columns which are not required for our analysis
news_data = news_data.drop(['date'], axis=1)


#dropping duplicates if any
news_data["text"] = news_data["text"].str.lower().str.strip()
news_data = news_data.drop_duplicates(
    subset=["text", "source", "category", "label"]
)


# ===============================
# FIX LABEL ENCODING (MANDATORY)
# ===============================
news_data["label"] = (
    news_data["label"]
    .str.lower()
    .map({"fake": 1, "real": 0})
)


print("Label distribution after encoding:")
print(news_data["label"].value_counts())


# -------- EDA 1: Fake vs Real news count --------

label_counts = news_data["label"].value_counts()

plt.figure(figsize=(6, 4))
label_counts.plot(kind="bar")
plt.title("Number of Fake vs Real News Articles")
plt.xlabel("News Type")
plt.ylabel("Number of Articles")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# -------- EDA 2: Number of news articles by source --------

source_counts = news_data["source"].value_counts().head(10)

plt.figure(figsize=(10, 5))
source_counts.plot(kind="bar")
plt.title("Top 10 News Sources by Number of Articles")
plt.xlabel("News Source")
plt.ylabel("Number of Articles")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# -------- EDA 3: Fake vs Real news by source --------

source_label_count = (
    news_data[news_data["source"].isin(source_counts.index)]
    .groupby(["source", "label"])
    .size()
    .unstack(fill_value=0)
)

source_label_count.plot(
    kind="bar",
    figsize=(12, 6),
    stacked=False
)

plt.title("Fake vs Real News Count by Source (Top 10 Sources)")
plt.xlabel("News Source")
plt.ylabel("Number of Articles")
plt.xticks(rotation=45)
plt.legend(fontsize=8, title="Label", title_fontsize=9)
plt.tight_layout()
plt.show()

# ===============================
# 3. training the model
# ===============================
news_data["content"] = (
    news_data["title"].fillna("") + " " +
    news_data["text"].fillna("")
)

X = news_data["content"]

y = news_data["label"]  # output (fake / real)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42, #to ensure the data remains the same across different runs
    stratify=y # to preserve the same proportion of fake and real news in both train and test sets
)

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.85,
    min_df=3,
    ngram_range=(1, 3),   # unigrams + bigrams + trigrams
    sublinear_tf=True
)



X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)



dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train_tfidf, y_train)

dummy_pred = dummy.predict(X_test_tfidf)

print("Dummy Accuracy:",
      accuracy_score(y_test, dummy_pred) * 100)


# ===============================
# Passive Aggressive Classifier (BEST for Fake News)
# ===============================

pa_model = PassiveAggressiveClassifier(
    max_iter=1000,
    C=0.75, # streangth of regularization
    random_state=42
)

pa_model.fit(X_train_tfidf, y_train)

pa_pred = pa_model.predict(X_test_tfidf)

print(f"Passive Aggressive Accuracy: "
      f"{accuracy_score(y_test, pa_pred) * 100:.2f}%")

print("\nClassification Report:\n")
print(classification_report(y_test, pa_pred))






def predict_fake_news(model, vectorizer):
    print("\n Fake News Detection System")
    user_text = input("Enter the news text:\n")

    user_text_tfidf = vectorizer.transform([user_text])

    prediction = model.predict(user_text_tfidf)[0]

    if prediction == 1 or prediction == "FAKE":
        print("\n This news is likely FAKE.")
    else:
        print("\n This news is likely REAL.")
predict_fake_news(pa_model, vectorizer)

