#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from pydantic.dataclasses import dataclass
from Neural_Style_Transfer import NeuralStyleTransfer
from tensorflow.keras.models import load_model
from numba import jit, cuda
import numpy as np 
import pickle
import json


# In[2]:


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# In[11]:


class model_input(BaseModel):
    
    content : list
    style : list
    iterations : int
    alpha : float = 2.5e-10
    beta : float = 3e-6
    gamma : float = 3e-6


# In[10]:


@app.post('/NST')
def stylize(input_parameters : model_input):
    
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    content = np.array(input_dictionary['content'], dtype=np.uint8)
    style = np.array(input_dictionary['style'], dtype=np.uint8)
    iterations = input_dictionary['iterations']
    alpha = input_dictionary['alpha']
    beta = input_dictionary['beta']
    gamma = input_dictionary['gamma']
    
    model = load_model("Model.h5")
    obj = NeuralStyleTransfer(model)
    
    @jit(target_backend='cuda')
    img = obj.run(iterations, content, style, alpha, beta, gamma)
    
    op = {'image': img.tolist()}
    

    return json.dumps(op)
    


# In[ ]:




