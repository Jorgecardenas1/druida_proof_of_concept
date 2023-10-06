# -*- coding: utf-8 -*-
from __future__ import print_function
from Utilities.ConvertImageToBinary import Binary
from pathlib import Path
import scipy
from scipy import ndimage
import glob
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os

#--------------------- Configuraciones varias 

#Directorio donde se encuentran los modelos entrenados
netGDir='C:/juan_pinto/DIE820/Proyecto_DIE820/Models/netG500.pt'  #En este caso se llama al que tiene 500 epocas de entrenamiento

#Directorio de datos de entrenamiento 
spectra_path = 'C:/juan_pinto/DIE820/Proyecto_DIE820/Training_Data/absorptionData_HybridGAN.csv'

#Se restauran las clases y variables 
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu               
        self.conv1 = nn.ConvTranspose2d(nz, ngf * 8, 6, 1, 0, bias=False)
        self.conv2 = nn.BatchNorm2d(ngf * 8)
        self.conv3 = nn.ReLU(True)
            # state size. (ngf*8) x 6 x 6
        self.conv4 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 6, 2, 2, bias=False)
        self.conv5 = nn.BatchNorm2d(ngf * 4)
        self.conv6 = nn.ReLU(True)
            # state size. (ngf*4) x 12 x 12
        self.conv7 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 6, 2, 4, bias=False)
        self.conv8 = nn.BatchNorm2d(ngf * 2)
        self.conv9 = nn.ReLU(True)
            # state size. (ngf*2) x 20 x 20
        self.conv10 = nn.ConvTranspose2d(ngf * 2, ngf, 6, 2, 5, bias=False)
        self.conv11 = nn.BatchNorm2d(ngf)
        self.conv12 = nn.ReLU(True)
            # state size. (ngf) x 34 x 34
        self.conv13 = nn.ConvTranspose2d(ngf, nc, 6, 2, 4, bias=False)
        self.conv14 = nn.Tanh()
            # state size. (nc) x 64 x 64
    
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

#--------------------- Para comparar lo generado por el modelo entrenado vs etiqueta real 

def compare(i, factor, shift, results_folder, netGDir, spectra_path):
            
    ##Carga el Generador
    netG = torch.load(netGDir, map_location='cpu')
    
    ##Crea las entradas para el generador 
    excelTestData = pd.read_csv(spectra_path, index_col = 0)
    
    #for z in range(shift):
        #excelTestData.insert(0,str(z),0)
    excelDataSpectra = excelTestData.iloc[:,:800]
    excelDataSpectra = excelDataSpectra.shift(shift,axis=1,fill_value=0)
    excelTestDataTensor = torch.tensor(factor*excelDataSpectra.values).type(torch.FloatTensor)
    testTensor = torch.Tensor()
    
    index = i
    tensor1 = torch.cat((excelTestDataTensor[index],torch.rand(400)))
    tensor2 = tensor1.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    tensor3 = tensor2.permute(1,0,2,3)
    testTensor = torch.cat((testTensor,tensor3),0)
    
    fake = netG(testTensor).detach().cpu()
    img_reshape = fake.permute(0,2,3,1)
    img = img_reshape.squeeze()
    img = img.numpy()
       
    excelTestDataNames = pd.read_csv(spectra_path)
    name = excelTestDataNames.iloc[index,0]
    print(name)      
    img = (img + 1)/2
    
    im_size = 64
    pmax = 4.0
    tmax = 10.0
    emax = 5.0
    
    psum = 0.0
    pnum = 0.0
    tsum = 0.0
    tnum = 0.0
    esum = 0.0
    enum = 0.0
    
    for row in range(im_size):
        for col in range(im_size):
            if img[row][col][0] > 0.2 or img[row][col][1] > 0.2:
                if img[row][col][0] > img[row][col][1]:
                    psum += img[row][col][0]
                    pnum += 1
                else:
                    esum += img[row][col][1]
                    enum += 1
            else:
                tsum += img[row][col][2]
                tnum += 1
    if pnum > 0:
        pAvg = psum / pnum
    else:
        pAvg = 0.0
        
    if tnum > 0:
        tAvg = tsum / tnum
    else:
        tAvg = 0.0
        
    if enum > 0:
        eAvg = esum / enum
    else:
        eAvg = 0.0
        
    
    pfake = pmax * pAvg
    tfake = tmax * tAvg
    efake = emax * eAvg
    
    preal = excelTestData.iloc[index, 801]
    treal = excelTestData.iloc[index, 802]
    ereal = excelTestData.iloc[index, 803]
            
    plt.imshow(img)
    plt.imsave(results_folder+ '/Results/' + str(i) + '-test.png',img)
    
    if pnum > enum:
        pindexfake = pfake
        pindexreal = preal
        classifier = 0
    else:
        pindexfake = efake
        pindexreal = ereal
        classifier = 1
    print("Fake Plasma/Index:", pindexfake)
    print("Real Plasma/Index:", pindexreal)
    print("Fake Thickness:", tfake)
    print("Real Thickness:", treal)
    
    return [pindexfake, pindexreal, tfake, treal, classifier]

#Solicita el indice de la base de datos para generar un fake a partir de ese modelo de antena y orientación
indices = [0, 20, 50, 800, 14000, 18000]
# indices = []
# for index in range(0,63):
#     indices.append(1 * index)
results_folder = os.path.dirname(os.path.realpath(__file__))
Path(results_folder+ '/Results').mkdir(parents=True, exist_ok=True) #ref: https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory-in-python
file = open(results_folder + '/Results/properties.txt',"w")
file.write("Index FakePlasma/Index RealPlasma/Index FakeThickness RealThickness Class(MIM=0/DM=1)")
for i in indices:
    props = compare(i, 1, 0, results_folder, netGDir, spectra_path)
    props.insert(0, i)
    row = ""
    for j in props:
        row += str(round(j, 2)) + " "
    file.write("\n" + row)
file.close()

#Convierte las imagenes en escala de grises 
im_size = 64
im_path = results_folder+ '/Results/*-test.png'
imgFolder = glob.glob(im_path)
imgFolder.sort()

for img in imgFolder:
    rgb = mpimg.imread(img)
    for row in range(im_size):
        for col in range(im_size):
            if rgb[row][col][0] > rgb[row][col][2] or rgb[row][col][1] > rgb[row][col][2]:
                rgb[row][col][0] = 0
                rgb[row][col][1] = 0
                rgb[row][col][2] = 0
            else:
                rgb[row][col][0] = 1
                rgb[row][col][1] = 1
                rgb[row][col][2] = 1
                
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    cv2.normalize(gray, gray, -1, 1, cv2.NORM_MINMAX)
    
    #Aplicando filtro gaussiano a la iamgen
    img_filter = scipy.ndimage.gaussian_filter(gray,sigma=0.75)
    ret, img_filter = cv2.threshold(img_filter,0.1,1,cv2.THRESH_BINARY) # 0 = negro, 1 = blanco; e
    
    plt.imshow(img_filter, cmap = "gray")
    plt.imsave(img[:-4]+'-bw.png', img_filter, cmap = "gray")

#Convertir imagen en b/n a binario para Lumerical    
Binary.convert(results_folder)

