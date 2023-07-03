import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump

data=pd.read_csv(r'C:\Users\Sanjay\Jupyter notebook\toxi classi\jigsaw-toxic-comment-train.csv')
print(data.shape)
print(data.head())

print(data.isnull().any())

def clean_text(text):
    text=text.lower()
    text=re.sub(r"what's","what is", text)
    text=re.sub(r"\ 's", " ", text)
    text=re.sub(r"\ 've", "have ", text)
    text=re.sub(r"can't", "cannot ", text)
    text=re.sub(r"n't", "not ", text)
    text=re.sub(r"i'm", "i am ", text)
    text=re.sub(r"\ 're", "are ", text)
    text=re.sub(r"\ 'd", "would ", text)
    text=re.sub(r"\ 'll", "will ", text)
    text=re.sub(r"\ 'scuse", "excuse ", text)
    text=re.sub('\W', ' ', text)
    text=re.sub('\s+', ' ', text)
    text=text.strip(' ')
    return text

data['comment_text']=data['comment_text'].map(lambda com : clean_text(com))

labels = data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
text=data['comment_text']

tfidf = TfidfVectorizer(max_features=5000)
features = tfidf.fit_transform(text)

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)  # You can adjust the number of estimators
rf_classifier.fit(x_train, y_train)

dump(rf_classifier, 'random_forest_model.joblib')
dump(tfidf, 'rf_model_tfidf.joblib')
#rf_classifier = load('random_forest_model.joblib')

y_pred = rf_classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)

print(accuracy)
print(classification_report)