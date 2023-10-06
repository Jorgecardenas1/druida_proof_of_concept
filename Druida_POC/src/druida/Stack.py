from .DataManager import datamanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

  
class Trainer:
    

    def __init__(self, learning_rate:float, batch_size:int, epochs:int):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

    def training(self, trainFunction,testFunction, train_dataloader, test_dataloader, model, loss_fn, optimizer):
        acc=0
        acc_test=0
        loss=0
        test_loss=0
        loss_values = []
        test_loss_values = []
        train_acc_hist = []
        test_acc = []

        for t in range(self.epochs):
            dataiter = iter(train_dataloader)
            testdataiter = iter(test_dataloader)


            print(f"Epoch {t+1}\n-------------------------------")
            acc,loss=trainFunction(next(dataiter), model, loss_fn, optimizer, t,acc,loss)
            acc_test,test_loss=testFunction(next(testdataiter), model, loss_fn,len(train_dataloader), t, acc_test,test_loss)

            loss_values.append(loss)
            test_loss_values.append(test_loss)
            train_acc_hist.append(acc)
            test_acc.append(acc_test)
        print("Done!")

        return loss_values,test_loss_values,train_acc_hist, test_acc
  
    
class DNN(nn.Module):

    def __init__(self, layers):
        super(DNN,self).__init__()

        self.checkDevice()
        self.layers = layers
        self.architecture=nn.Sequential()

        for layer in layers:
            self.architecture.add_module(layer['name'],layer['layer'])

        

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

        #{"name":"hidden1","layer":nn.Linear(8,120), "type":"hidden", "index",0}
        self.output =  input
        for layer in self.layers:
            action=layer['layer']
            self.output = action(self.output)
                
                
                
        return self.output
    

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
