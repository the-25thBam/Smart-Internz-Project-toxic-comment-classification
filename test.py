from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer

rf_classifier = load('random_forest_model.joblib')
tfidf=load('rf_model_tfidf.joblib')

new_comment="I'm going to freakin kill you"

new_comment_feature = tfidf.transform([new_comment])

# Make the prediction
prediction = rf_classifier.predict(new_comment_feature)

l = prediction.flatten()
l1 = l.tolist()
l2 = []

if l1[0] == 1:
    l2.append("toxic")
if l1[1] == 1:
    l2.append("sever_toxic")
if l1[2] == 1:
    l2.append("obscene")
if l1[3] == 1:
    l2.append("threat")
if l1[4] == 1:
    l2.append("insult")
if l1[5] == 1:
    l2.append("identity_hate")

# Print the prediction
print(f"Comment: {new_comment}")
print(f"Predicted Labels: {l2}")