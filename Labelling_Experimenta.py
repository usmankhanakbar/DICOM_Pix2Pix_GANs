from numpy import load
from matplotlib import pyplot
from tensorflow.keras.backend import clear_session
from nn.GAN_5 import GAN_5 as GAN
from tensorflow.keras.utils import plot_model
import numpy as np
from keras.utils.vis_utils import plot_model
from numpy import save
import os

 # Import all training and test images and their target derivatives... 
 # Importing them in their Saggital, 
def loadNpy(fn):
    images_trg_s = load(fn+'/images_trg_s.npy')
    images_drv_s = load(fn+'/images_drv_s.npy')
    images_trg_c = load(fn+'/images_trg_c.npy')
    images_drv_c = load(fn+'/images_drv_c.npy')
    images_trg_t = load(fn+'/images_trg_t.npy')
    images_drv_t = load(fn+'/images_drv_t.npy')
    
    images_test_s = load(fn+'/images_test_s.npy')
    images_t_drv_s = load(fn+'/images_t_drv_s.npy')
    images_test_c = load(fn+'/images_test_c.npy')
    images_t_drv_c = load(fn+'/images_t_drv_c.npy')
    images_test_t = load(fn+'/images_test_t.npy')
    images_t_drv_t = load(fn+'/images_t_drv_t.npy')
        
    return images_trg_s,images_drv_s,images_trg_c,images_drv_c,images_trg_t,images_drv_t,images_test_s,images_t_drv_s,images_test_c,images_t_drv_c,images_test_t,images_t_drv_t


def combineImgs(images_trg_s,images_drv_s,images_trg_c,images_drv_c,images_trg_t,images_drv_t,images_test_s,images_t_drv_s,images_test_c,images_t_drv_c,images_test_t,images_t_drv_t):
    
    trgImages = np.vstack((images_trg_s,images_trg_c,images_trg_t))
   
    trgDrvs = np.vstack((images_drv_s,images_drv_c,images_drv_t))
    
    testImages = np.vstack((images_test_s,images_test_c,images_test_t)) 
   
    testDrvs = np.vstack((images_t_drv_s,images_t_drv_c,images_t_drv_t))
    return trgImages,trgDrvs,testImages,testDrvs

def loadNSaveImgs():

    fn = r'C:\Users\usman\OneDrive\Documents\Thesis_Stuff\NumpyArraysForProject\trg_data'
    directory = r'C:\Users\usman\OneDrive\Documents\Thesis_Stuff\NumpyArraysForProject\trgNtestData'
    images_trg_s,images_drv_s,images_trg_c,images_drv_c,images_trg_t,images_drv_t,images_test_s,images_t_drv_s,images_test_c,images_t_drv_c,images_test_t,images_t_drv_t = loadNpy(fn)
    print('Step 1 Complete...')
    trgImages,trgDrvs,testImages,testDrvs = combineImgs(images_trg_s,images_drv_s,images_trg_c,images_drv_c,images_trg_t,images_drv_t,images_test_s,images_t_drv_s,images_test_c,images_t_drv_c,images_test_t,images_t_drv_t)
    print('Step 2 Complete...')
    save(directory+'/trgImages.npy',trgImages)
    save(directory+'/trgDrvs.npy',trgDrvs)
    save(directory+'/testImages.npy',testImages)
    save(directory+'/testDrvs.npy',testDrvs)
    print('Training and Test Images Saved in Directory')

def loadNpyImgs():
    directory = r'C:\Users\usman\OneDrive\Documents\Thesis_Stuff\NumpyArraysForProject\trgNtestData'
    return load(directory+'/trgImages.npy'),load(directory+'/trgDrvs.npy'),load(directory+'/testImages.npy'),load(directory+'/testDrvs.npy')
#########################################################################################################################################################################################################################################
# Generative Adversial Model Definitions from Here
# https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/ 
# Code Help and reference from this website
def preProcessImg(nparray):
    nparray = np.asarray(nparray)
    mask = np.where(np.isnan(nparray), 0, nparray)
    return mask



def trainModel():
    
    trgImages,trgDrvs,testImages,testDrvs = loadNpyImgs()
    print('All trainning and test Images as List loaded')
    
    n_samples = 1
    image_shape = (trgImages.shape[1],trgImages.shape[2],trgImages.shape[0])
    filename = r'C:\Users\usman\OneDrive\Documents\Thesis_Stuff\NumpyArraysForProject\trgNtestData'
    trgImages = trgImages.reshape(image_shape)
    trgDrvs = trgDrvs.reshape(image_shape)
    trgIm =[]
    trgDr = []
    for i in range(trgImages.shape[2]):
        trgIm.append( preProcessImg(trgImages[:,:,i]))
        trgDr.append(preProcessImg(trgDrvs[:,:,i]))
    trgImages = np.asarray(trgIm)
    trgDrvs = np.asarray(trgDr)
    trgImages = np.reshape(trgImages,(trgImages.shape[1],trgImages.shape[2],trgImages.shape[0]))
    trgDrvs = np.reshape(trgDrvs,(trgDrvs.shape[1],trgDrvs.shape[2],trgDrvs.shape[0]))
    clear_session()
    g = GAN(trgImages,trgDrvs,n_samples,image_shape, filename)
    # defining models
    d_model = g.define_discriminator()
    #plot_model(d_model, to_file=filename+'/model_plot_Discriminator_2.png', show_shapes=True, show_layer_names=True)
    plot_model(d_model, to_file=filename+'/model_plot_Discriminator_5.png', show_shapes=True, show_layer_names=True)
    g_model = g.define_generator()
    plot_model(g_model, to_file=filename+'/model_plot_Generator_5.png', show_shapes=True, show_layer_names=True)
   
    # define composite model
    gan_model = g.define_gan(g_model,d_model)
    plot_model(gan_model, to_file=filename+'/model_plot_GAN_5.png', show_shapes=True, show_layer_names=True)
    
    #train model
    g.train(d_model,g_model,gan_model)

#########################################################################################################################################################################################################################################
if __name__ == '__main__':
    #loadNSaveImgs()
    trainModel()
   
    
    
    