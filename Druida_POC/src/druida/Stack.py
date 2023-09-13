from .DataManager import datamanager

from .setup import inputType
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.Module):

    def __init__(self, layers):
        super(DNN,self).__init__()

        self.checkDevice()
        self.layers = layers
        self.architecture=nn.Sequential()

        for layer in layers:
            self.architecture.add_module(list(layer.keys())[0],layer[list(layer.keys())[0]])

        

    def push(self, layer):
        self.layers.append(layer) #each layer must come as dictionary with nn type
        return self.layers
 
    def drop_last(self):
        self.layers.pop() #each layer must come as dictionary with nn type
        return self.layers

    def clear(self):
        self.layers.clear() #each layer must come as dictionary with nn type
        return self.layers
    
    def checkDevice(self):
        self.device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
        )
    
    def forward(self, input):

        self.output =  input
        for value in self.layers:
            layer=value[list(value.keys())[0]]
            self.output = layer(self.output)
                
        return self.output
    


    #If we want to use different types of entries as vector
    def input_modeling(self, object):
            
            self.input_type = object['type']
            self.inputData=object['data']

            if inputType['image'] == self.input_type:
                flatten = nn.Flatten()
                self.flat_image = flatten(self.inputData)
                return self.flat_image
            
            elif inputType['vector'] == self.input_type:
                
                #Este es un ejemplo randomizando la entrada
                self.inputTensor=torch.rand(object['size'], dtype=object['torchType'], device=object['device'])
                return self.inputTensor
            else:
                pass






class Generator:
    def __init__(self, layers, neuron_matrix, weights, input_layer):
        self.layers = layers
        self.neuron_matrix = neuron_matrix
        self.weights=weights
        self.input_layer =  input_layer

    def create(self):
        stack=datamanager.dataSets("other name",2,3)
        stack.create()


class Discriminator:
    def __init__(self, layers, neuron_matrix, weights, input_layer):
        self.layers = layers
        self.neuron_matrix = neuron_matrix
        self.weights=weights
        self.input_layer =  input_layer

    def create(self):
        stack=datamanager.dataSets("other name",2,3)
        stack.create()
