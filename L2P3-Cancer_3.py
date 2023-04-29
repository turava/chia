#!/usr/bin/env python
# coding: utf-8

# In[110]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import Model


# In[106]:


#Cargar los datos
new_data = pd.read_csv('cancer_dataset.csv')


# In[107]:


# Elimina la primera columna (identificador del paciente)
new_data = new_data.drop('id', axis=1)


# In[108]:


# Codifica las etiquetas de diagnóstico (B = 0, M = 1)
label_encoder = LabelEncoder()
new_data['diagnosis'] = label_encoder.fit_transform(new_data['diagnosis'])


# In[109]:


# Realiza predicciones para los nuevos datos
predictions = model.predict(new_data.drop('diagnosis', axis=1))


# In[83]:


# Obtener la clase predicha para cada fila
predicted_classes = np.argmax(predictions, axis=1)

# Decodificar las clases predichas (0 = benigno, 1 = maligno)
decoded_classes = label_encoder.inverse_transform(predicted_classes)


# In[ ]:




