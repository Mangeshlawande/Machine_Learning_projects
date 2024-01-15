import uvicorn
from fastapi import FastAPI
from cls import var_data
import pickle

app = FastAPI()
pickle_in = open("C:/Users/mange/OneDrive/Desktop/Data_Science Batch Noob to pro max/✅ Machine Learning/14)November/ML_Projects/P5 clustering/clusters.pkl","rb")
classifier = pickle.load(pickle_in)

@app.get("/")
def index():
    return {'Deployment':"Implementing KMeans algorithm it can able to detect the respective datapoint belong to which clusters;"}

@app.post('/predict')
def predict(data:var_data):  
    data = data.dict()
    X1 = data["X1"]
    X2 = data["X2"]

    prediction = classifier.predict([[X1,X2]])
    if(prediction[0]==0):
        prediction = "The Datapoint belongs to First Cluster !!"
    elif(prediction[0]==1):
        prediction = "The datapoint belongs to Second Cluster !!"
    return {
        'prediction': prediction
    }

if __name__=='__main__':
    uvicorn.run(app,host="127.0.0.1",port=7000)


