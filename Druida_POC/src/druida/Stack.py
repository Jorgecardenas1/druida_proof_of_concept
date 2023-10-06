from .DataManager import datamanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

  
class Trainer:
    

    def __init__(self, learning_rate:float, batch_size:int, epochs:int, workers=0, gpu_number=0):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.workers = workers
        self.gpu_number=gpu_number
        
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

    def multiGPU(self, network):
        print('available Device:'+network.device)
        if (network.device == 'cuda' and (self.gpu_number > 1)):
            network=nn.DataParallel(network,list(range(self.gpu_number)))
  
    
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
    



class Generator(nn.Module):
    def __init__(self, ngpu, input_size, mapping_size, channels ):
        super(Generator, self).__init__()
        

        self.checkDevice()

        self.ngpu = ngpu            

        self.conv1 = nn.ConvTranspose2d(input_size, mapping_size * 8, kernel_size=6, stride=1, padding=0, bias=False)
        self.conv2 = nn.BatchNorm2d(num_features=mapping_size * 8)
        self.conv3 = nn.ReLU(True)
        self.conv4 = nn.ConvTranspose2d(mapping_size * 8, mapping_size * 4, 6, 2, 2, bias=False)
        self.conv5 = nn.BatchNorm2d(mapping_size * 4)
        self.conv6 = nn.ReLU(True)
        self.conv7 = nn.ConvTranspose2d(mapping_size * 4, mapping_size * 2, 6, 2, 4, bias=False)
        self.conv8 = nn.BatchNorm2d(mapping_size * 2)
        self.conv9 = nn.ReLU(True)
        self.conv10 = nn.ConvTranspose2d(mapping_size * 2, mapping_size, 6, 2, 5, bias=False)
        self.conv11 = nn.BatchNorm2d(mapping_size)
        self.conv12 = nn.ReLU(True)
        self.conv13 = nn.ConvTranspose2d(mapping_size, channels, 6, 2, 4, bias=False)
        self.conv14 = nn.Tanh()

    def forward(self, input):
        imageOut = input
        imageOut = self.conv1(imageOut)
        imageOut = self.conv2(imageOut)
        imageOut = self.conv3(imageOut)
        imageOut = self.conv4(imageOut)
        imageOut = self.conv5(imageOut)
        imageOut = self.conv6(imageOut)
        imageOut = self.conv7(imageOut)
        imageOut = self.conv8(imageOut)
        imageOut = self.conv9(imageOut)
        imageOut = self.conv10(imageOut)
        imageOut = self.conv11(imageOut)
        imageOut = self.conv12(imageOut)
        imageOut = self.conv13(imageOut)
        imageOut = self.conv14(imageOut)               
        return imageOut

    def checkDevice(self):
        self.device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
        )
    


class Discriminator(nn.Module):
    def __init__(self, ngpu=0, image_size=32, discriminator_mapping_size=0, channels=3):
        super(Discriminator, self).__init__()

        
        self.checkDevice()

        self.ngpu = ngpu            
        self.image_size = image_size
        self.channels = channels


        self.l1 = nn.Linear(800, image_size*image_size*channels, bias=False)           
        self.conv1 = nn.Conv2d(2*channels, discriminator_mapping_size, 6, 2, 4, bias=False) 
        self.conv2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(discriminator_mapping_size, discriminator_mapping_size * 2, 6, 2, 5, bias=False)
        self.conv4 = nn.BatchNorm2d(discriminator_mapping_size * 2)
        self.conv5 = nn.LeakyReLU(0.2, inplace=True)
        self.conv6 = nn.Conv2d(discriminator_mapping_size * 2, discriminator_mapping_size * 4, 6, 2, 4, bias=False)
        self.conv7 = nn.BatchNorm2d(discriminator_mapping_size * 4)
        self.conv8 = nn.LeakyReLU(0.2, inplace=True)
        self.conv9 = nn.Conv2d(discriminator_mapping_size * 4, discriminator_mapping_size * 8, 6, 2, 2, bias=False)
        self.conv10 = nn.BatchNorm2d(discriminator_mapping_size * 8)
        self.conv11 = nn.LeakyReLU(0.2, inplace=True)
        self.conv12 = nn.Conv2d(discriminator_mapping_size * 8, 1, 6, 1, 0, bias=False)
        self.conv13 = nn.Sigmoid()


    def forward(self, input, label, b_size):
        x1 = input
        x2 = self.l1(label)
        x2 = x2.reshape(int(b_size/self.ngpu),self.channels,self.image_size,self.image_size) 
        combine = torch.cat((x1,x2),dim=1) # concatenate in a given dimension
        combine = self.conv1(combine)
        combine = self.conv2(combine)
        combine = self.conv3(combine)
        combine = self.conv4(combine)
        combine = self.conv5(combine)
        combine = self.conv6(combine)
        combine = self.conv7(combine)
        combine = self.conv8(combine)
        combine = self.conv9(combine)
        combine = self.conv10(combine)
        combine = self.conv11(combine)
        combine = self.conv12(combine)
        combine = self.conv13(combine)
        return combine

    def checkDevice(self):
        self.device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
        )
    