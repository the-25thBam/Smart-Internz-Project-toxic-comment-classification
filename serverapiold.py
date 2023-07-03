import numpy as np

from joblib import load

'''
loaded_model = tf.keras.models.load_model('vectoriser_model')
vectoriser = loaded_model.layers[0]

model = tf.keras.models.load_model('toxicity.h5')
cols_target = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult','identity_hate']
'''

rf_classifier = load('random_forest_model.joblib')
tfidf=load('rf_model_tfidf.joblib')
cols_target = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult','identity_hate']

from flask import Flask,render_template,request

app=Flask(__name__)
@app.route("/")
def indexPage():
    return render_template('index.html')

@app.route("/submit", methods=['GET', 'POST'])
def submiting():
    reqCols = []
    comm = request.form.get('comment')
    inp = tfidf.transform([comm])
    #inp=vectoriser(comm)
    #predArr=(rf_classifier.predict(np.expand_dims(inp,0))>0.5)
    predArr = (rf_classifier.predict(inp)>0.5)
    #predArr=(model.predict(np.expand_dims(inp,0))>0.5)
    for i in range(0,len(predArr[0])):
        if(predArr[0][i]):
            reqCols.append(cols_target[i])
    if(len(reqCols)==0):
        return 'the comment is clean'
    return 'the comment is: ' + str(reqCols)

if __name__ == '__main__':
    app.run(debug=True)







