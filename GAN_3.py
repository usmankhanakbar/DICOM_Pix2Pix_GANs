from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from matplotlib import pyplot
from time import sleep
from matplotlib.animation import FuncAnimation
from tensorflow.keras.models import model_from_json
import numpy as np

from tensorflow.math import log
from tensorflow import ones_like, equal
from tensorflow.math import multiply as mul
import tensorflow as tf
from tensorflow.keras.backend import clip
from tensorflow.keras.backend import epsilon
from tensorflow import map_fn

class GAN_3:
    def __init__(self,trgData,trgDrvs,n_samples,image_shape, filename ):
        self.trgData = trgData
        self.trgDrvs = trgDrvs
        self.n_samples = n_samples
        self.image_shape = image_shape
        self.filename = filename
    
    def weighted_bin_cross_entropy(self,w1,w2):
        def loss(y_true,y_pred):
            y_pred = clip(y_pred,epsilon(),1-epsilon())
            ones = ones_like(y_true)
            msk = equal(y_true,ones)
            res,_ = map_fn(lambda x: (mul(-log(x[0]),w1) if x[1] is True else mul(-log(1-x[0]),w2),x[1]),(y_pred,msk),dtype=(tf.float32,tf.bool))
            return res
        return loss
        
    def define_discriminator(self):
        # Weight initialization
        init = RandomNormal(stddev=0.02)
        # source image input
        im_shape = (self.image_shape[0],self.image_shape[1],1)
        in_src_image = Input(shape=im_shape)
        # target image input
        in_target_image= Input(shape=im_shape)
        # Concatenate images channel wise
        merged = Concatenate()([in_src_image,in_target_image])
        # C64
        d = Conv2D(64,(4,4),strides=(2,2),padding='same',kernel_initializer = init)(merged)
        d = LeakyReLU(alpha=0.2)(d)
        # C128
        d = Conv2D(128,(2,2),strides=(1,1),padding = 'same',kernel_initializer = init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha = 0.2)(d)
        # C128
        d = Conv2D(128,(4,4),strides = (2,2), padding = 'same',kernel_initializer = init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # C256
        d = Conv2D(256,(4,4),strides = (2,2),padding ='same',kernel_initializer = init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha = 0.2)(d)
        # C256
        d = Conv2D(256,(2,2),strides=(1,1),padding = 'same',kernel_initializer = init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha = 0.2)(d)
        # second Last Output Layer
        d = Conv2D(512,(4,4),padding = 'same',kernel_initializer = init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha = 0.2)(d)
        # patch output
        d = Conv2D(1,(4,4),padding='same',kernel_initializer= init)(d)
        patch_out = Activation('sigmoid')(d)
        # define model
        model =  Model([in_src_image,in_target_image],patch_out)
        #compile Model
        opt = Adam(lr=0.0002,beta_1 = 0.5)
       # model.compile(loss = 'binary_crossentropy',optimizer = opt, loss_weights = [0.5])
        model.compile(loss = self.weighted_bin_cross_entropy(0.3,0.7),optimizer = opt)
        return model
    
    # define an encoder block
    def define_encoder_block(self,layer_in,n_filters,filtersize = 2, stride = 1,batchnorm = True):
        # weight initialization
        init = RandomNormal(stddev = 0.02)
        # add downsampling layer
        g = Conv2D(n_filters,(filtersize,filtersize),strides = (stride,stride),padding='same',kernel_initializer=init)(layer_in)
        # conditionallyy add batch normalization
        if batchnorm:
            g = BatchNormalization()(g,training = True)
        # leaky relu activation
        g = LeakyReLU(alpha = 0.2)(g)
        return g
    
    # define a decoder block
    def decoder_block(self,layer_in,skip_in,n_filters,filtersize = 2, stride = 1,dropout = True):
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # add upsampling layer
        g = Conv2DTranspose(n_filters,(filtersize,filtersize),strides = (stride,stride),padding = 'same',kernel_initializer=init)(layer_in)
        # add batch normalization
        g = BatchNormalization()(g,training = True)
        # conditionally add dropout
        if dropout:
            g = Dropout(0.5)(g,training = True)
        # merge with skip connection
        g = Concatenate()([g, skip_in])
        # relu activation
        g = Activation('relu')(g)
        return g
    #define the standalone generator model
    def define_generator(self):
        #weight initialization
        
        init = RandomNormal(stddev=0.02)
        # image input 
        im_shape = (self.image_shape[0],self.image_shape[1],1)
        in_image = Input(shape = im_shape)
        # encoder model
        e1 = self.define_encoder_block(in_image,64,batchnorm = False)
        e2 = self.define_encoder_block(e1,128)
        e3 = self.define_encoder_block(e2,128)
        e4 = self.define_encoder_block(e3,256,filtersize = 2, stride = 1)
        e5 = self.define_encoder_block(e4,256,filtersize = 2, stride = 1)
        e6 = self.define_encoder_block(e5,512,filtersize = 4, stride = 2)
        e7 = self.define_encoder_block(e6,512,filtersize = 4, stride = 2)
        
        # bottleneck, no batch norm and relu
        bb = Conv2D(512,(4,4),strides = (2,2), padding = 'same',kernel_initializer=init)(e7)
        bb = Activation('relu')(bb)
        # decoder model
        d1 = self.decoder_block(bb,e7,512,filtersize = 4, stride = 2)
        d2 = self.decoder_block(d1,e6,512,filtersize = 4, stride = 2)
        d3 = self.decoder_block(d2,e5,256,filtersize = 4, stride = 2)
        d4 = self.decoder_block(d3, e4, 256,filtersize = 2, stride = 1, dropout = False)
        d5 = self.decoder_block(d4, e3, 128,filtersize = 2, stride = 1, dropout = False)
        d6 = self.decoder_block(d5, e2, 128,filtersize = 2, stride = 1, dropout = False)
        d7 = self.decoder_block(d6, e1, 64,filtersize = 2, stride = 1, dropout = False)
        
        # output
        g = Conv2DTranspose(1,(2,2),strides = (1,1), padding = 'same',kernel_initializer = init)(d7)
        out_image = Activation('tanh')(g)
        # define model
        model = Model(in_image,out_image)
        return model
    
    def define_gan(self,g_model,d_model):
        # make weights in the discriminator not trainable
        for layer in d_model.layers:
            if not isinstance(layer, BatchNormalization):
                layer.trainable = False
        # define the source imae
        im_shape = (self.image_shape[0],self.image_shape[1],1)
        in_src = Input(shape= im_shape)
        # connect the source image to generator input
        gen_out = g_model(in_src)
        # connect the source input and generator output to the discriminator input
        dis_out = d_model([in_src,gen_out])
        # src image as input, generated image and classification output
        model = Model(in_src,[dis_out, gen_out])
        # compile Model
        opt = Adam(lr = 0.0002, beta_1 = 0.5)
        model.compile(loss = ['binary_crossentropy','mae'],optimizer = opt, loss_weights = [1,100])
        #plot_model(model, show_shapes=True, to_file=self.filename+'/multiple_vgg_blocks.png')
        return model
    
   
    
    # select a batch of random samples, returns images and target
    def generate_real_samples(self,patch_shape):
        trainA, trainB = self.trgData, self.trgDrvs
        ix = randint(0,trainA.shape[2],self.n_samples)
        # retrieve selected images
        X1,X2 = trainA[:,:,ix], trainB[:,:,ix]
        # Recale the intensities to - 1 to +1 for Tanh activation
        X1 = (X1-np.min(X1))/(np.max(X1)-np.min(X1)) * (1-(-1)) - 1
        X2 = (X2-np.min(X2))/(np.max(X2)-np.min(X2)) * (1-(-1)) - 1
        # generate 'real' class labels (1)
        
        y = ones((self.n_samples, patch_shape,patch_shape,1))
        return [X1,X2], y
    
    # generate a batch of images, returns images and targets
    def generate_fake_samples(self,g_model,sample,patch_shape):
        # generate fake instances
        X = g_model.predict(sample)
        # create fake class labels (0)
        y = zeros((len(X), patch_shape, patch_shape,1))
        return X, y
    # generate samples and 
    def animate(self,i,d_loss1,d_loss2,g_loss):
        
        
        pyplot.cla()
        pyplot.plot(i,d_loss1,label='d_loss1')
        pyplot.plot(i,d_loss2,label='d_loss2')
        pyplot.plot(i,g_loss,label='g loss')
   
        pyplot.legend(loc = 'upper left')
        pyplot.tight_layout()
    def summarize_performance(self,step, g_model,patch):
        [X_realA, X_realB],_ = self.generate_real_samples(patch)
        X_realA = np.expand_dims(X_realA,axis=0)
        X_realB = np.expand_dims(X_realB,axis=0)
        X_fakeB,_ = self.generate_fake_samples(g_model,X_realA,patch)
        
        #X_realA = (X_realA-np.min(X_realA))/(np.max(X_realA)-np.min(X_realA))*255.0
        
        X_realA = (X_realA+1)/2.0
        X_realB = (X_realB+1)/2.0
        X_fakeB = (X_fakeB+1)/2.0
        
        X_realA = X_realA*255
        X_realB = X_realB*255
        X_fakeB = X_fakeB*255
        
        # fig, axs = pyplot.subplots(1,3)
        # for i in range(self.n_samples):
           
        #     axs[0].imshow(X_realA[0,:,:,0])
        
        # for i in range(self.n_samples):
            
        #     axs[1].imshow(X_fakeB[0,:,:,0])
        
        # for i in range(self.n_samples):
            
        #     axs[2].imshow(X_realB[0,:,:,0])
    
        # save plot to file
        # filename1 = 'plot_%06d.png' %(step+1)
        # pyplot.savefig(self.filename +'/images/'+ filename1)
        # pyplot.close()
        
        # save the generator model
        filename2 = 'model_%06d.json'%(step+1)
        filename3 = 'model_%06d.h5'%(step+1)
        # g_model.save(self.filename +'/model/'+filename2)
        
        # save the GAN Model
        model_json = g_model.to_json()
        with open(self.filename+'/'+filename2,"w") as json_file:
            json_file.write(model_json)
    
        #serialize wts  to HD5
        g_model.save_weights(self.filename+"/"+filename3)
        print('>Saved : %s and %s '% (filename2, filename3))
        print("Saved model to disk")
        
        x_vals = []
        y1_vals = []
        y2_vals = []
        y3_vals = []
        
    def train(self,d_model,g_model,gan_model,n_epochs=100,n_batch=1):
        # determine the output square shape of the discriminator
        n_patch = d_model.output_shape[1]
        # unpack dataset
        trainA, trainB = self.trgData,self.trgDrvs
        # calculate the number of batches per training epoch
        bat_per_epo = int(len(trainA)/n_batch)
        # calculate the number of training iterations
        n_steps = bat_per_epo * n_epochs
        # manually enumerate epocs
        x_vals  = []
        y1_vals = []
        y2_vals = []
        y3_vals = []
        for i in range(n_steps):
            [X_realA, X_realB], y_real = self.generate_real_samples(n_patch)
            X_realA = np.expand_dims(X_realA,axis=0)
            X_realB = np.expand_dims(X_realB,axis=0)
            X_fakeB, y_fake = self.generate_fake_samples(g_model,X_realA,n_patch)
            d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
            d_loss2 = d_model.train_on_batch([X_realA,X_fakeB], y_fake)
            g_loss,_,_ = gan_model.train_on_batch(X_realA,[y_real,X_realB])
            print('>%d,d1[%.3f] d2[%.3f] g[%.3f]' %(i+1,d_loss1,d_loss2,g_loss))
            x_vals.append(i+1)
            y1_vals.append(d_loss1)
            y2_vals.append(d_loss2)
            y3_vals.append(g_loss)
            np.savetxt(self.filename+'/'+'output_g3.csv', [np.asarray(x_vals),np.asarray(y1_vals),np.asarray(y2_vals),np.asarray(y3_vals)], delimiter=',', fmt='%f')
       
            # FuncAnimation(pyplot.gcf(),self.animate(x_vals,y1_vals,y2_vals,y3_vals),interval = 1000)
            
            # pyplot.tight_layout()
            # pyplot.show()
            
            # sleep(0.1)
            # pyplot.pause(0.0001)
            # summarize model performance
            if (i+1)%(bat_per_epo * 10) == 0:
                self.summarize_performance(i,g_model,n_patch)
        
    
    
    
    
    