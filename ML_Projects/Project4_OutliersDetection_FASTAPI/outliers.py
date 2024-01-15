import uvicorn
from fastapi import FastAPI
from meta import var_data
import pickle
app = FastAPI()
pickle_in = open("C:/Users/mange/OneDrive/Desktop/Data_Science Batch Noob to pro max/✅ Machine Learning/14)November/ML_Projects/P4/outlier.pkl","rb")
classifier=pickle.load(pickle_in)

@app.get('/')
def index():
    return {'Deployment': 'The ML algorithm implementation to detect inliers and outliers in the given data'}

@app.post('/predict')
def predict(data:var_data):
    data = data.dict()
    X1=data['X1']
    X2=data['X2']

    prediction = classifier.predict([[X1,X2]])
    if(prediction[0] == 1):
        prediction="It's an Inlier"
    elif(prediction[0] == -1):
        prediction=" It's an Outlier"
    return {
        'prediction': prediction
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=5000)
