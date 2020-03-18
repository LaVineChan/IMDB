import sys
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer as TF
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import cross_val_score

#preprocess the text
def clean_text(origin_text):
    # delete html tag
    text = BeautifulSoup(origin_text).get_text()
    # Remove punctuation and illegal characters
    text = re.sub("[^a-zA-Z]", " ", text)
    # Convert all characters to lowercase, and perform word segmentation through space characters
    words = text.lower().split()
    # remove stopwords
    stop_words = set(stopwords.words("english"))
    meaningful_words = [w for w in words if w not in stop_words]
    # back to str
    cleaned_text = " ".join(meaningful_words)
    return cleaned_text

train_df = pd.read_json('./data/train.json', encoding='utf-8')
print(train_df['review_text'])
test_df = pd.read_json('./data/test.json', encoding='utf-8')

train_df['text'] = train_df['review_text'].apply(lambda x: clean_text(x))
test_df['text'] = test_df['review_text'].apply(lambda x: clean_text(x))
print(train_df['text'].head())
train_df['is_spoiler'] = train_df['is_spoiler'].apply(lambda x: 1 if x else 0)

train_df = train_df.sample(frac=1).reset_index(drop=True)
tfidf = TF(
    analyzer="word",
    tokenizer=None,
    preprocessor=None,
    stop_words=None,
    max_features=200)

print("Creating the tfidf vector...\n")
tfidf.fit(train_df['text'])
x_train = tfidf.transform(train_df['text'])
x_train = x_train.toarray()

x_test = tfidf.transform(test_df['text'])
x_test = x_test.toarray()

print(x_train.shape)
print(x_test.shape)

y_train = train_df['is_spoiler']
print(y_train.value_counts())

model = LR(solver='liblinear')
model.fit(x_train, y_train)
print("10折交叉验证：")
print(np.mean(cross_val_score(model, x_train, y_train, cv=10, scoring="accuracy")))

#submit
preds = model.predict(x_test)
submission = pd.DataFrame({'id': range(len(preds)), 'pred': preds})
submission['id'] = submission['id']
submission.to_csv("./data/ml_submission.csv", index=False, header=False)
print(submission.head(5))