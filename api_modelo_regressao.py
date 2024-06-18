from pydantic import BaseModel #usada para criar modelos de dados com validação de tipo.
from fastapi import FastAPI # que é usada para criar a aplicação API.
import uvicorn #Importa o servidor ASGI uvicorn, que permite rodar sua aplicação FastAPI.
import joblib  #  Importa a biblioteca joblib, usada para carregar modelos de machine learning salvos.


# Criar uma instância do FastaApi
app = FastAPI()

# Criar um classe que terá os dados do requesty vody para a API
class request_body(BaseModel):
    horas_estudo : float

# Carregar modelo para realizar a predição
modelo_pontuacao =  joblib.load('./modelo_regressao.pkl')

@app.post("/predict")
def predict(data : request_body): 
    # Preparar os dados para predição
 input_feature = [[data.horas_estudo]]

 # Realizar a predição
 y_pred = modelo_pontuacao.predict(input_feature)[0].astype(int)# pegar indice 0

 return {'pontuacao_test' : y_pred.tolist()}