
from .DataManager import datamanager
import numpy as np

class DNN_Stack:
    def __init__(self, layers, neuron_matrix, weights, input_layer):
        self.layers = layers
        self.neuron_matrix = neuron_matrix
        self.weights=weights
        self.input_layer =  input_layer

    def create(self):
        stack=datamanager.dataSets("other name",2,3)
        stack.create()
