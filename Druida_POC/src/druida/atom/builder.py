
from ..DataManager import datamanager

class Stack:
    def __init__(self, layers, scattererArray):
        self.layers = layers
        self.scattererArray = scattererArray

    def create(self):
        pass

class Scatterer:
    def __init__(self, name, image_matrix, size):
        self.name = name
        self.image_matrix = image_matrix
        self.size=size

    def create(self):
        pass