import pandas as pd
import re
import string
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

df = pd.read_csv('Dataset/spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Data Cleaning and Preprocessing
def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    return text

df['message'] = df['message'].apply(preprocess_text)

# Split the dataset
X = df['message']
y = df['label']

#80% is given for training and validation while 20% is given for testing
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# second time 70% is for training and 10% is for validation
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.125, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_val_vectorized = vectorizer.transform(X_val)
X_test_vectorized = vectorizer.transform(X_test)

# Models used
classifiers = {
    'Naive Bayes': MultinomialNB(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'KNN': KNeighborsClassifier()
}

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, clf in classifiers.items():
    clf.fit(X_train_vectorized, y_train)
    y_pred_val = clf.predict(X_val_vectorized)

    scores = cross_val_score(clf, X_train_vectorized, y_train, cv=kf, scoring='accuracy')

    print(f"Results for {name}:")
    print(f"Cross-Validation Accuracy: {scores.mean():.2f} ± {scores.std():.2f}")
    print(f"Validation Accuracy: {accuracy_score(y_val, y_pred_val):.2f}")
    print(f"Validation Precision: {precision_score(y_val, y_pred_val, pos_label='spam'):.2f}")
    print(f"Validation Recall: {recall_score(y_val, y_pred_val, pos_label='spam'):.2f}")
    print(classification_report(y_val, y_pred_val, target_names=['Genuine', 'spam']))
    print("="*60)

# Saving the models for future use
with open('models_and_vectorizer.pkl', 'wb') as f:
    pickle.dump((vectorizer, classifiers), f)
