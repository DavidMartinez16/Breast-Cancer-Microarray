# -*- coding: utf-8 -*-

"""
Autor: Juan David Martínez Gualdrón
Asignatura: Aprendizaje Automático
Fecha: 23/06/2020
Maestría en Ingeniería Electrónica
Facultad de Ingeniería Eléctrica y Electrónica
Título del proyecto :  Modelo de aprendizaje automático en datos de la expresión genética
                       para la clasificación y predicción del cáncer de pecho
                       
"""

# --------------- IMPORTO LAS LIBRERÍAS A UTILIZAR ---------------------
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler

# --------------- IMPORTO EL CONJUNTO DE DATOS ------------------------

data=pd.read_csv('Breast_GSE45827.csv')
print(data.head())

# --------------- ANÁLISIS DE LA DATA ---------------------------------

# Verifico si hay datos nulos o perdidos en el dataset
print('Datos nulos o perdidos en el conjunto de datos')
print(data.isnull().sum())

# Verifico los tipos de datos del dataset
print('Tipos de datos del dataset')
print(data.dtypes)

# Verifico la cantidad de datos o la forma del dataset
print('Cantidad de datos')
print(data.shape)

# Verifico la información de dataset
print('Información del conjunto de datos')
print(data.info())

# Descripción del dataset
print('Descripción del conjunto de datos')
description=data.describe()
print(description)

# Cantidad de muestras por clase
print('Número de instancias por clase')
print(data['type'].value_counts())

# ----------------- PREPROCESAMIENTO DE LOS DATOS ---------------------

# Elimino las columnas inncesarias para el analisis

# Se elimina la columna de muestras ya que no aporta información relevante
datac=data.drop(['samples'],axis=1)

# Obtener las dummies de cada clase de cancer
dum = pd.get_dummies(datac['type'])

# Concatenar los dummies con la data original
datac = pd.concat([dum,datac],axis=1)

# Reemplazar valores nulos por 0, por si acaso
datac.replace(np.nan,0,inplace=True)

# Guardar la data procesada
datac.to_csv('Data_Processed.csv', index_label=False)