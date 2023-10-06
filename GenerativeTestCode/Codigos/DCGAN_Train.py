
#---------------------
#---------------------
#---------------------
#--------------------- Cuando veas estos signos, significa que debes separarlo en otra pestañita
#--------------------- Iré describiendo que hace cada linea de codigo


# -*- coding: utf-8 -*-
from __future__ import print_function
from Utilities.SaveAnimation import Video
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from IPython.display import HTML
import time
import pandas as pd
import pickle


#----------------------Se comprueba si está conectado dispositivo CUDA

#Obtiene que tarjeta de video se está usando
print("CUDA is available: {}".format(torch.cuda.is_available()))
print("CUDA Device Count: {}".format(torch.cuda.device_count()))
print("CUDA Device Name: {}".format(torch.cuda.get_device_name(0)))

#Directorio donde se encuentra los datos de entrenamiento
spectra_path = 'C:/juan_pinto/DIE820/Proyecto_DIE820/Training_Data/absorptionData_HybridGAN.csv'

#Directorio donde se guardan los modelos (Generators and Discriminators)
save_dir = 'C:/juan_pinto/DIE820/Proyecto_DIE820/Models/'

#Directorio donde se encuentran las carpetas con clases
img_path = 'C:/juan_pinto/DIE820/Proyecto_DIE820/Training_Data/'


#--------------------- Chequeta el archivo csv 

def Excel_Tensor(spectra_path):
    # Location of excel data
    excelData = pd.read_csv(spectra_path, header = 0, index_col = 0)    
    excelDataSpectra = excelData.iloc[:,:800] #index until the last point of the spectra in the Excel file
    excelDataTensor = torch.tensor(excelDataSpectra.values).type(torch.FloatTensor)
    return excelData, excelDataSpectra, excelDataTensor

excelData, excelDataSpectra, excelDataTensor = Excel_Tensor(spectra_path)

#--------------------- Carga archivo log de entrenamiento y toma el tiempo

f = open('training_log.txt','w')
start_time = time.time()
local_time = time.ctime(start_time)
print('Start Time = %s' % local_time)
print('Start Time = %s' % local_time, file=f)


#Para no truncar el contenido del tensor
torch.set_printoptions(profile="full")

#La semilla 9 extrañamente fue la que no divergió al iniciar entrenamiento
manualSeed = 9
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

#--------------------- Configuración de la GAN

#Number of workers for dataloader (for Windows workers must = 0, for reference: https://github.com/pytorch/pytorch/issues/2341)
workers = 0 

#Tamaño del Batch
batch_size = 16

#Tamaño de las iamgenes de entrenamiento, todas se dejan de este tamaño usando Transformers.
image_size = 64

#Cantidad de canales que tienen las imagenes (son RGB codificadas, asi que son 3)
nc = 3 

#Tamaño del vector latente Z (i.e. Tamaño de la entrada del generador)
latent = 400
gan_input = excelDataTensor.size()[1] + latent

#Tamaño del mapeo de caracteristicas del generador
ngf = 128

#Tamaño del mapeo de caracteristicas para el discriminador
ndf = 64

#Numero de epocas de entrenamiento
num_epochs = 501

#Learning rate para los optimizadores
lr = 0.0001

#Beta1 hyperparam para optimizador Adam
beta1 = 0.5

#Numero de GPUS disponibles, si es 0 se hace con CPU.
ngpu = 1

#--------------------- Obtiene el nombre de los archivos para las clases

#Crea el dataset. Usa "dataset.imgs" para mostrar el nombre del archivo
dataset = dset.ImageFolder(root=img_path,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize([0.5],[0.5]) 
                           ]))
#Crea el dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=workers)

#Decide en que dispositivo se entrenará (prefiere siempre GPU) 
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

#Inicialización de los pesos para el Generador (NetG) y para el Discriminador (NetD)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


#--------------------- Se define la arquitectura de las redes

#Se define la arquitectura de la red del Generador
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu            
        self.conv1 = nn.ConvTranspose2d(gan_input, ngf * 8, 6, 1, 0, bias=False)
        self.conv2 = nn.BatchNorm2d(ngf * 8)
        self.conv3 = nn.ReLU(True)
        self.conv4 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 6, 2, 2, bias=False)
        self.conv5 = nn.BatchNorm2d(ngf * 4)
        self.conv6 = nn.ReLU(True)
        self.conv7 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 6, 2, 4, bias=False)
        self.conv8 = nn.BatchNorm2d(ngf * 2)
        self.conv9 = nn.ReLU(True)
        self.conv10 = nn.ConvTranspose2d(ngf * 2, ngf, 6, 2, 5, bias=False)
        self.conv11 = nn.BatchNorm2d(ngf)
        self.conv12 = nn.ReLU(True)
        self.conv13 = nn.ConvTranspose2d(ngf, nc, 6, 2, 4, bias=False)
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

#Se crea el generador 
netG = Generator(ngpu).to(device)

#PAra el manejo de multi-GPU 
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

#Aplica la función para inicilaizar los pesos par que tengan promedio=0 y sigma_2 = 0.2 
netG.apply(weights_init)

#Se muestra el modelo
print(netG)

#--------------------- Se genera la arquitectura para el Discriminador 

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.l1 = nn.Linear(800, image_size*image_size*nc, bias=False)           
        self.conv1 = nn.Conv2d(2*nc, ndf, 6, 2, 4, bias=False) 
        self.conv2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(ndf, ndf * 2, 6, 2, 5, bias=False)
        self.conv4 = nn.BatchNorm2d(ndf * 2)
        self.conv5 = nn.LeakyReLU(0.2, inplace=True)
        self.conv6 = nn.Conv2d(ndf * 2, ndf * 4, 6, 2, 4, bias=False)
        self.conv7 = nn.BatchNorm2d(ndf * 4)
        self.conv8 = nn.LeakyReLU(0.2, inplace=True)
        self.conv9 = nn.Conv2d(ndf * 4, ndf * 8, 6, 2, 2, bias=False)
        self.conv10 = nn.BatchNorm2d(ndf * 8)
        self.conv11 = nn.LeakyReLU(0.2, inplace=True)
        self.conv12 = nn.Conv2d(ndf * 8, 1, 6, 1, 0, bias=False)
        self.conv13 = nn.Sigmoid()

    def forward(self, input, label):
        x1 = input
        x2 = self.l1(label)
        x2 = x2.reshape(int(b_size/ngpu),nc,image_size,image_size) 
        combine = torch.cat((x1,x2),1)
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

#Crea el Discriminador 
netD = Discriminator(ngpu).to(device)

#Para manejar multiples GPU 
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

#Aplica la función para inicilaizar los pesos par que tengan promedio=0 y sigma_2 = 0.2 
netD.apply(weights_init)

#Se muestra el modelo 
print(netD)


#--------------------- Se configura la interaccion entre NetG y NetD 

#Inicia la función BCELoss

criterion = nn.BCELoss()


#Crea el batch del vector latente, en donde se visualizará la progresion del Generador 
testTensor = torch.Tensor()
for i in range (100):
    fixed_noise1 = torch.cat((excelDataTensor[i*int(np.floor(len(excelDataSpectra)/100))],torch.rand(latent)))
    fixed_noise2 = fixed_noise1.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    fixed_noise = fixed_noise2.permute(1,0,2,3)
    testTensor = torch.cat((testTensor,fixed_noise),0)
testTensor = testTensor.to(device)

#Establecer convención para etiquetas reales y falsas durante el entrenamiento
real_label = random.uniform(0.9,1.0)
fake_label = 0

#Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


#--------------------- Se inicia el loop de entrenamiento 

##Training Loop
#Listas para seguir progreso del entrenamiento 
img_list = [] #Para el video de animación
G_losses = [] #Para mostrar las perdidas del Generador
D_losses = [] #Para mostrar las perdidas del Discriminador
iters = 0
noise = torch.Tensor()
noise2 = torch.Tensor()
print("Starting Training Loop...")

#Por cada epoca
x=0
for epoch in range(num_epochs):
    x=0
    # Por cada batch en el dataloader
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Actualizar NetD: maximizar log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Entrena con el batch con todas las etiquetas verdaderas
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)

        # Genera el batch del espectro, vectores latentes, and propiedades     
        for j in range(batch_size):
            excelIndex = x*batch_size+j
            try:
                gotdata = excelDataTensor[excelIndex]
            except IndexError:
                break
            tensorA = excelDataTensor[excelIndex].view(1,800)
            noise2 = torch.cat((noise2,tensorA),0)      
            
            tensor1 = torch.cat((excelDataTensor[excelIndex],torch.rand(latent)))
            tensor2 = tensor1.unsqueeze(1).unsqueeze(1).unsqueeze(1)         
            tensor3 = tensor2.permute(1,0,2,3)
            noise = torch.cat((noise,tensor3),0)         
                              
        noise = noise.to(device)            
        noise2 = noise2.to(device)                
        
         # Forward pass del batch real a través de NetD
        output = netD.forward(real_cpu,noise2).view(-1)
        # Calcula la perdida de all-real batch
        errD_real = criterion(output, label)
        # Calcula el gradients para NetD en backward pass
        errD_real.backward()
        D_x = output.mean().item()
              
        ## Entrenamiento con all-fake batch                
        # Genera un batch de imagenes falsas con NetG
        fake = netG.forward(noise)
        label.fill_(fake_label)
        # Clasifica todos los batch falsos con NetD
        output = netD.forward(fake.detach(),noise2).view(-1)
        # Calcula la perdida de NetD durante el btach de imagenes falsas
        errD_fake = criterion(output, label)
        # Calcula el gradiente para este batch 
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Se suman los gradientes de los batch all-real y all-fake 
        errD = errD_real + errD_fake
        # Se actualiza NetD con la optimizacion
        optimizerD.step()

        ############################
        # (2) Actualizar NetG: maximizar log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  #las etiquetas de las imagenes galsas son reales para el generador 
        # Mientras se actualiza NetD, se hacer un forward pass en el lote de all-fake de NetD
        output = netD.forward(fake,noise2).view(-1)
        # Calcula la perdida de NetG basandose en este output
        errG = criterion(output, label)
        # Calcula los gradientes de NetG 
        errG.backward()
        D_G_z2 = output.mean().item()
        # Actualiza NetG 
        optimizerG.step()

        # Estadisticas del entrenamiento 
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2), file=f)

        # Guarda las perdidas para ser ploteadas 
        G_losses.append(errG.item())
        D_losses.append(errD.item())

       #  Revisa si el Generador guarda las salidas de NetG en fixed_noise 
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(testTensor).detach().cpu()
            img_list.append(vutils.make_grid(fake, nrow=10, padding=2, normalize=True))

        iters += 1
        noise = torch.Tensor()
        noise2 = torch.Tensor()     
        x += 1
    if epoch % 50 == 0:
        ##Guarda el modelo en el directorio cada 50 epocas 
        torch.save(netG, save_dir + 'netG' + str(epoch) + '.pt')
        torch.save(netD, save_dir + 'netD' + str(epoch) + '.pt')

#--------------------- Guarda lo hecho en el entrenamiento y muestra en pantalla los datos de seguimiento de progreso 

#Para ver cuanto tiempo tomó entrenar todo
local_time = time.ctime(time.time())
print('End Time = %s' % local_time)
print('End Time = %s' % local_time, file=f)
run_time = (time.time()-start_time)/3600
print('Total Time Lapsed = %s Hours' % run_time)
print('Total Time Lapsed = %s Hours' % run_time, file=f)
f.close()


#Crea el video con el progreso del Generador 
ims, ani = Video.save_video(save_dir, img_list, G_losses, D_losses)


#Plotea y guarda las perdidas de  NetG y NetD. 
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="Generator Loss")
plt.plot(D_losses,label="Discriminator Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('losses.png')
plt.show()