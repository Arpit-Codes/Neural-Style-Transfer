#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing all the necessary libraries
from tensorflow.keras.applications import vgg19
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
import PIL


# In[3]:


class NeuralStyleTransfer:
    
    def __init__(self, model):
        self.h, self.w, self.c = (600,600,3)
        
        #loading vgg19 model and creating feature extractor
        
        
        #creating SGD optimizer
        self.optimizer = SGD(ExponentialDecay(initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96))
        self.feature_extractor = model 


        #defining activation layers for content and style
        self.content_layers = ['block5_conv2']

        self.style_layers = ['block1_conv1',
                        'block2_conv1',
                        'block3_conv1',
                        'block4_conv1',
                        'block5_conv1'] 
    
    #extracting features using vgg19
    def extract_feature(self, content, style):
        self.content_feature = self.feature_extractor(content)
        self.style_feature = self.feature_extractor(style)
        
    #utility function to preprocess images  
    def preprocess(self, img):

        img = cv2.resize(img, (self.w,self.h))
        img = np.expand_dims(img, axis=0)
        img = vgg19.preprocess_input(img)
        img = tf.convert_to_tensor(img)

        return img
    
    def restore(self, img):
        img = img.reshape((self.h, self.w, self.c))
        img[:, :, 0] = img[:, :, 0]+103.939
        img[:, :, 1] = img[:, :, 1]+116.779
        img[:, :, 2] = img[:, :, 2]+123.68
        img = img[:,:,::-1]
        # print(type(img))
        img = np.clip(img, 0,255).astype(np.uint8)  
        return img
    
    #utility function to calculate content loss
    def content_loss(self, content, generated):
        return tf.reduce_sum(tf.square(generated-content))
    
    #utility function to calculate gram matrix
    def gram_matrix(self, x):
        result = tf.linalg.einsum('bijc,bijd->bcd', x, x)
        return result
    
    #utility function to calculate style loss
    def style_loss(self, style, generated):
        s = self.gram_matrix(style)
        g = self.gram_matrix(generated)
        size = self.h*self.w
        channel = self.c
        return tf.reduce_sum((tf.square(s-g))/(4*(size**2)*(channel**2)))
    
    #utility function to calculate variation loss
    def total_variation_loss(self, x):
        a = x[:, :, 1:, :] - x[:, :, :-1, :]
        b = x[:, 1:, :, :] - x[:, :-1, :, :]
        return tf.reduce_sum(tf.abs(a)) + tf.reduce_sum(tf.abs(b))
    
    #utility function to calculate total loss
    def total_loss(self, content, style, generated, alpha, beta, gamma):
        
        self.extract_feature(content, style)
        generated_feature = self.feature_extractor(generated)

        total_loss = tf.zeros(shape=())

        for layer in self.content_layers:
        # print(layer)
            content_activation = self.content_feature[layer]
            generated_activation = generated_feature[layer]
            total_loss += (alpha)*(self.content_loss(content_activation, generated_activation))/(len(self.content_layers))
        # print(content_activation.shape)




        for layer in self.style_layers:
            style_activation = self.style_feature[layer]
            generated_activation = generated_feature[layer]
            total_loss += (beta)*(self.style_loss(style_activation, generated_activation))/(len(self.style_layers))

            total_loss += gamma*(self.total_variation_loss(generated))


        return total_loss

    #utility function to calculate gradient
    @tf.function
    def compute_gradient(self, content, style, generated, alpha, beta, gamma):
        with tf.GradientTape() as tape:
            loss = self.total_loss(content, style, generated, alpha, beta, gamma)
        grad = tape.gradient(loss, generated)
        return loss, grad
    
    def run(self, iterations, content, style, alpha, beta, gamma):
        
        content = self.preprocess(content)
        style = self.preprocess(style)
        generated = tf.Variable(content) 
        
        for i in range(iterations):
            loss, grad = self.compute_gradient(content, style, generated, alpha, beta, gamma)
            self.optimizer.apply_gradients([(grad, generated)])
            
        return self.restore(generated.numpy())


# In[ ]:




