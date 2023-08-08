from fastapi import FastAPI
import pandas as pd 
import pickle
from pydantic import BaseModel

class Juego(BaseModel):
    publisher: str
    release_year: int
    early_access: bool
    metascore: float
    Indie: bool

app = FastAPI()

df_juegos =pd.read_csv('steam_limpio2.csv')

with open('fitted_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get('/genero')
async def genero(year: int):
    df_filtrado = df_juegos[df_juegos['release_year'] == year]
    df_filtrado['genres'] = df_filtrado['genres'].str.replace('\'', '').str.replace('[', '').str.replace(']', '')
    df_generos = df_filtrado['genres'].str.split(', ', expand=True).stack().value_counts().head(5)
    return dict(zip(df_generos.index.tolist(), df_generos.tolist()))

@app.get('/specs')
async def specs(year: int):
    df_filtrado = df_juegos[df_juegos['release_year'] == year]
    df_filtrado['specs']= df_filtrado['specs'].str.replace('\'', '').str.replace('[','').str.replace(']','')
    df_specs = df_filtrado['specs'].str.split(', ', expand=True).stack().value_counts().head(5)
    return dict(zip(df_specs.index.tolist(), df_specs.tolist()))

@app.get('/earlyaccess')
async def earlyaccess(year: int):
    df_filtrado = df_juegos[(df_juegos['release_year'] == year) & (df_juegos['early_access'] == True)]
    return len(df_filtrado)

@app.get('/juegos')
async def juegos(year: str):
    juegos_lanzados = df_juegos.loc[df_juegos['release_year'] == int(year), 'app_name'].tolist()
    return juegos_lanzados

@app.get('/sentiment')
async def sentiment(year: str):
    df_filtrado = df_juegos.loc[df_juegos['release_year'] == int(year)]
    conteo_sentimientos = df_filtrado['sentiment'].value_counts().to_dict()
    return conteo_sentimientos

@app.get('/metascore')
async def metascore(year: str):
    df_filtrado = df_juegos.loc[df_juegos['release_year'] == int(year)]
    df_ordenado = df_filtrado.sort_values(by=['metascore'], ascending=False)
    top_5_juegos = df_ordenado.head(5).set_index('app_name')['metascore'].to_dict()
    return top_5_juegos

@app.get('/predict')
async def predict(publisher: str, release_year: int, early_access: bool, metascore: float, Indie: bool):
    # Convertir los datos de entrada en un arreglo de características
    X = [[publisher, release_year, early_access, metascore, Indie]]
    # Hacer la predicción con el modelo cargado
    precio_predicho = model.predict(X)[0]
    # Devolver la predicción como respuesta
    return {'precio_predicho': precio_predicho}