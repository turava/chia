#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# In[3]:


#Cargar los datos
data = pd.read_csv('cancer_dataset.csv')

# Dividir el conjunto de datos en entrenamiento y prueba
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Separar las características de la clase
train_X = train_data.iloc[:, 2:]
train_Y = train_data.iloc[:, 1]

test_X = test_data.iloc[:, 2:]
test_Y = test_data.iloc[:, 1]

# Crear el modelo de perceptrón multicapa
model = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=1000)

# Entrenar el modelo
model.fit(train_X, train_Y)

# Predecir la clase para el conjunto de prueba
predictions = model.predict(test_X)

# Evaluar la precisión del modelo
accuracy = accuracy_score(test_Y, predictions)
print("Precisión del modelo: {:.2f}".format(accuracy))


# In[ ]:




