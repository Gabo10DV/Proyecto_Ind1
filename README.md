# Proyecto_Ind1

##### El siguiente proyeccto consiste en crear un API en base a un datset de juegos de steam dicha api debe de tener varias funciones, entre ellas mostar los juegos lanzados en un año, el top de mejores juegos segun el meta score, los generos as comunes y un modelo de machine learning 

## Limpieza de datos 
##### Al data set se le realizo un proceso ETL y EDA con el fin de tener datos mas prolijos con los que trabajar, entro lo hecho a los datos podemos encontrar:
- Esndarizar datos
- cambiar tipos de datos
- crear y eliminar columnas 
- rellenar o elinimar filas en blanco
##### para ver mas del proceso puede ver el link de [limpieza de datos](https://colab.research.google.com/drive/1siu6N9wbr-sEOTReYLzBTbchRKhVfE3L?usp=sharing "limpieza de datos")

## Creacion de modelo ML
##### el objetivo de este modelo es predecir el precio de un juego, para esto se uso el data set ya limpio para buscar la relacion entre las caracteristicas de los juegos, la cual podemos ver en el siguiente cuadro:
[![cuadro de realcion ](https://tinypic.host/image/WhatsApp-Image-2023-08-03-at-2.51.28-PM.GO27R "cuadro de realcion ")](https://tinypic.host/image/WhatsApp-Image-2023-08-03-at-2.51.28-PM.GO27R "cuadro de realcion ")
##### lamentablemente los datos proporcionados no aportan mucha informacion sobre el precio por lo que se tomo a criterio personal que caraccterirsticas podrian afectar el precio de un juegos las cuales fueron:
- publishe
- release_year
- early_access
- metascore
- Indie

##### teniendo estos datos se procedio a crear el modelo 
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    
    Y = juegos['price']
    X = juegos[['publisher','release_year', 'early_access', 'metascore', 'Indie']]
    
    model = LinearRegression()
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    
    print(model.score(X_test, Y_test))
    print(mean_absolute_error(Y_test, predictions))

##### este modelo dio con un porcentaje de predicion de 31% y con un error de 6.02
##### pudes ver mas del proceso en [la creacion del modelo ](https://colab.research.google.com/drive/1v8qxq1R62wCb1gfDUlZQpCzk_RSAkL4i?usp=sharing "la creacion del modelo ")

## La API y sus funciones
##### para este proyecto se uso la libreria de FastAPI, uvicorn y panda; tambien se uso pickle para poder importar facilmente el modelo ml ya creado 
##### los endpoints tiene como funcion :
1. genero: devuelve un diccionario de los cinco géneros de juegos más comunes y cuantos juegos estan en ese genero para un año determinado.
2. specs: devuelve las cinco especificaciones más comunes para un año determinado.
3.  juegos: retorna todos los juegos del año especificado
4.  earlyacces: indica cuantos juegos tubieron un acceso temprano
5. sentiment: retorna un diccionario de las oponiones de los usarios sean mixtas o muy buenas y cuantos juegos estan por cada tipo de oponion 
##### toma en consideracion que los juegos se dividieron se la sigueinte manera:
    def get_sentiment(row):
        if row['metascore'] <= 70:
            return 'mixed'
        elif row['metascore'] <= 80:
            return 'mostly Positive'
        elif row['metascore'] <= 86:
            return 'positive'
        else:
            return 'very positive'
    df_juegos['sentiment'] = df_juegos.apply(get_sentiment, axis=1)

6. metascore: al ingresar un año regresa un top como diccionario de los mejores 5 juegos y cual fue su score 