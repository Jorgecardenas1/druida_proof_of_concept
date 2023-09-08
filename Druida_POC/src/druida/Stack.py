
from .DataManager import datamanager
import numpy as np

class Stack:
    def __init__(self, layers, neuron_matrix, weights, input_layer):
        self.layers = layers
        self.neuron_matrix = neuron_matrix
        self.weights=weights
        self.input_layer =  input_layer

    def create(self):
        stack=datamanager.dataSets("other name",2,3)
        stack.create()

class MetaSurface:

    def __init__(self, size, grid_size, grid_atom):
        self.size = size
        self.grid_size = grid_size
        self.grid_atom=grid_atom

    def create(self):
        stack=datamanager.dataSets("other name",2,3)
        stack.create()

class DNN:
    def __init__(self, layers, neuron_matrix, weights, input_layer):
        self.layers = layers
        self.neuron_matrix = neuron_matrix
        self.weights=weights
        self.input_layer =  input_layer

    def create(self):
        stack=datamanager.dataSets("other name",2,3)
        stack.create()

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
