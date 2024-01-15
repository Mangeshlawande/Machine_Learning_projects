import uvicorn
from fastapi import FastAPI
from para import house_data
import pickle
app = FastAPI()
pickle_in = open("C:/Users/mange/OneDrive\Desktop/Data_Science Batch Noob to pro max/✅ Machine Learning/14)November/ML_Projects/P3/regressor.pkl","rb")
model=pickle.load(pickle_in)

@app.get('/')
def index():
    return {'Deployment': 'The ML algorithm which give the predicted value of house price according to the given data;'}

@app.post('/predict')
def predict(data:house_data):
    data = data.dict()
    MedInc=data['MedInc']
    HouseAge=data['HouseAge']
    AveRooms=data['AveRooms']
    AveBedrms=data['AveBedrms']
    Population=data['Population']
    AveOccup=data['AveOccup']
    Latitude=data['Latitude']
    Longitude=data['Longitude']


    prediction = model.predict([[MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude]])
    
    return {
        'prediction': prediction[0]
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)