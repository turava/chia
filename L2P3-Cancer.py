#!/usr/bin/env python
# coding: utf-8

# In[65]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical


# In[66]:


data = pd.read_csv("cancer_dataset.csv")

# Elimina la primera columna (identificador del paciente)
data = data.drop('id', axis=1)

# Codifica las etiquetas de diagnóstico (B = 0, M = 1)
label_encoder = LabelEncoder()
data['diagnosis'] = label_encoder.fit_transform(data['diagnosis'])


# In[67]:


train_data, predict_data = train_test_split(data, test_size=0.2, random_state=42)


# In[68]:


model = Sequential()
model.add(Dense(64, activation='relu', input_dim=30))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='sigmoid'))


# In[69]:


# Compila el modelo
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])


# In[70]:


# Entrena el modelo
model.fit(train_data.drop('diagnosis', axis=1), to_categorical(train_data['diagnosis']),
          epochs=100, batch_size=32, validation_split=0.2)


# In[ ]:





# In[ ]:




