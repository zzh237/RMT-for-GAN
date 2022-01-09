from abc import ABC, abstractmethod
from analysis.args import *
# from pt_concgantools import * 
import numpy as np 

# import torch
# from torch import nn
# import torchvision.transforms as tfs
# from torch.utils.data import DataLoader
# from torchvision.datasets import MNIST
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from tensorflow.keras.layers import LeakyReLU,BatchNormalization, Embedding, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential, Model
from tensorflow.keras.models import model_from_json
from tensorflow.keras import backend as K
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
# from google.colab import drive
# drive.mount('/content/gdrive')
# path = 'gdrive/My Drive/'


class KerasGAN(ABC):
    @abstractmethod
    def __init__(self, args, subname, cnn=None):
        self.args = args
        self.subname = subname 
        # self.cnn = cnn 
        self.construct_path()
        
        if os.path.isfile("{}/epochs.txt".format(self.model_path)):
            file = open("{}/epochs.txt".format(self.model_path),'r')
            epochstr = file.readline()
            self.epochs_start = int(epochstr)
        else:
            self.epochs_start = 1 

        self.latent_dim = 100
        self.norm_coef = 127.5
        self.train_data = self.load_data()
        x = self.train_data[0]
        shape = x.shape
        if x.ndim == 3:
            self.img_rows = shape[1]
            self.img_cols = shape[2]
            self.img_shape = (self.img_rows, self.img_cols)
        
            # self.channels = 1
            # x = np.expand_dims(x, axis=3)
            # self.train_data[0] = x 
        elif x.ndim == 4:
            self.img_rows = shape[1]
            self.img_cols = shape[2]
            self.channels = shape[3]
            self.img_shape = (self.img_rows, self.img_cols, self.channels)
        
        else:
            pass 
        
        self.classes = self.args.classes
        self.num_classes = len(self.classes)

        self.batch_size = self.args.keras_gan['batch_size']
        self.epochs = self.args.keras_gan['epochs']
        self.sample_interval = self.args.keras_gan['sample_interval']

        self.optimizer = Adam(0.0002, 0.5)

    @abstractmethod
    def build_logic(self):
        pass 

    @abstractmethod
    def build_generator(self):
        pass 

    @abstractmethod
    def build_discriminator(self):
        pass 
    
    @abstractmethod
    def train(self):
        pass 

    @property
    def exp_n(self):
        return self._exp_n
    @exp_n.setter
    def exp_n(self, value):
        self._exp_n = value 


    def construct_path(self):
        self.model_path = os.path.join('model',self.subname,'mnist',self.args.run)
        if os.path.exists(self.model_path) == False:
            os.makedirs(self.model_path)
        
        # self.img_dir = os.path.join('images','simpleGAN')
        # if not os.path.exists(self.img_dir):
        #     os.makedirs(self.img_dir)

    def load_data(self):
        (x_train, y_train), (_, _) = mnist.load_data()
        ## save 50 real imgs to local for check
        if self.epochs_start == 1:
            self.save_as_images(x_train[0:60], y_train[0:60], file_name='images/{}/mnist/{}/real_images/real'.format(self.subname, self.args.run))
        x_train = (x_train.astype(np.float32) - self.norm_coef)/self.norm_coef 
        # The activation function of the output layer of the generator is tanh, which returns a value between -1 and 1. 
        # To scale that to 0 and 255 (which are the values you expect for an image), 
        # we have to multiply it by 127.5 (so that -1 becomes -127.5, and 1 becomes 127.5), 
        # and then add 127.5 (so that -127.5 becomes 0, and 127.5 becomes 255).
        #  We then have to do the inverse of this when feeding an image into the discriminator 
        #  (which will expect a value between -1 and 1).
        # Convert shape from (60000, 28, 28) to (60000, 784)
        # x_train = x_train.reshape(60000, 784)

        return [x_train, y_train]

    def draw_images(self, examples=25, dim=(5,5), figsize=(10,10), file_name='training_img'):
        dir = '/'.join(file_name.split('/')[:-1])
        if not os.path.exists(dir):
            os.makedirs(dir)

        noise= np.random.normal(loc=0, scale=1, size=[examples, 100])
        if self.subname == 'SimpleCGAN':
            classes = np.asarray([i%self.num_classes for i in range(examples)])
            classes = classes.reshape((-1,1)) 
            
            generated_images = self.generator.predict([noise,classes])
        elif self.subname == 'SimpleGAN':
            generated_images = self.generator.predict(noise)
        # generated_images = generated_images.reshape(25,28,28) #reshape now but could do in the generator
        plt.figure(figsize=figsize)
        for i in range(generated_images.shape[0]):
            plt.subplot(dim[0], dim[1], i+1)
            plt.imshow(generated_images[i], interpolation='nearest', cmap='Greys')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(file_name)

    def save_GAN(self, epoch):
        # serialize model to JSON
        # model_json = model.to_json()
        # with open(os.path.join(self.model_path, "model.json"), "w") as json_file:
        #     json_file.write(model_json)
        # serialize weights to HDF5
        # model.save_weights(os.path.join(self.model_path, "model.h5"))
        self.generator.save_weights(os.path.join(self.model_path, "G_model.h5"),True)
        self.discriminator.save_weights(os.path.join(self.model_path, "D_model.h5"),True)
        file = open("{}/{}".format(self.model_path,'epochs.txt'),'w')
        file.write('{}'.format(epoch))
        file.close()
        print("Saved model to disk")
    
    def load_GAN(self):
        # load json and create model
        json_file = open(os.path.join(self.model_path, 'model.json'), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(os.path.jion(self.model_path,"model.h5"))
        print("Loaded model from disk")

    
    def generate_images(self):
        
        all_noise_vectors = []
        all_images = []
        for i, label in enumerate(self.classes):
            ## check if image exist or not:  
            file_name='images/{}/mnist/{}/generated_images/{}/fake'.format(self.subname, self.args.run, self.exp_n)
            exist_flag = False
            filename_numbers = list(range(self.args.num_imgs_per_class)) 
            for k in [filename_numbers[0],filename_numbers[-1]]:
                current_file_name = file_name + '%s_%d.png' % (label, k)
                if os.path.isfile(current_file_name):
                    exist_flag = True
                else:
                    exist_flag = False
            if exist_flag == True:
                print("Images of class: %s (%d/%d) already existed" % (label, i + 1, len(self.classes)))
                continue 
            
            print("Generating images of class: %s (%d/%d)" % (label, i + 1, len(self.classes)))
            noise_vectors, images,labeldatas = self.generate_batch_images(label)
            for i in range(self.args.num_imgs_per_class // self.batch_size - 1):
                noise_vec, imgs,labeldata = self.generate_batch_images(label)
                images = np.concatenate((images, imgs), 0)
                noise_vectors = np.concatenate((noise_vectors, noise_vec), axis=0)
                if labeldatas is not None: 
                    labeldatas = np.concatenate((labeldatas, labeldata), 0)
            
            if self.args.represent_way == 'image':
                #save images by label to the local directory
                self.save_as_images(images, labeldatas, file_name='images/{}/mnist/{}/generated_images/{}/fake'.format(self.subname, self.args.run, self.exp_n))

            all_images.append(images) # a list has num of classes elements
        # return all_images
                
            
            # all_images.append(images)
            # all_noise_vectors.append(noise_vectors)
        # return all_noise_vectors, all_images

    @abstractmethod
    def generate_batch_images(self, *args):
        
        pass 

    def convert_to_images(self, obj):
        """ Convert an output tensor from BigGAN in a list of images.
            Params:
                obj: tensor or numpy array of shape (batch_size, channels, height, width)
            Output:
                list of Pillow Images of size (height, width)
        """
        try:
            import PIL
        except ImportError:
            raise ImportError("Please install Pillow to use images: pip install Pillow")

        # if not isinstance(obj, np.ndarray):
        #     obj = obj.detach().numpy()

        # obj = obj.transpose((0, 2, 3, 1))
        # obj = np.clip(((obj + 1) / 2.0) * 256, 0, 255)

        img = []
        for i, out in enumerate(obj):
            out_array = np.asarray(np.uint8(out), dtype=np.uint8)
            img.append(PIL.Image.fromarray(out_array))
        return img


    def save_as_images(self, obj, labels, file_name='output'):
        """ Convert and save an output tensor from BigGAN in a list of saved images.
            Params:
                obj: tensor or numpy array of shape (batch_size, channels, height, width)
                file_name: path and beggingin of filename to save.
                    Images will be saved as `file_name_{image_number}.png`
        """
        if isinstance(labels, np.ndarray):
            labels = labels.reshape((-1,))
        
        img = self.convert_to_images(obj)
        dir = '/'.join(file_name.split('/')[:-1])
        if not os.path.exists(dir):
            os.makedirs(dir)
        for (i, out), label in zip(enumerate(img),labels):
            current_file_name = file_name + '%s_%d.png' % (label, i)
            print("Saving image to {}".format(current_file_name))
            out.save(current_file_name, 'png')
            
        # # gen = gen.reshape(-1,self.img_rows, self.img_cols)
        # print(gen.shape)
        # ###############################################################
        # #直接可视化生成图片
        # if save:
        #     for i in range(0,len(gen)):
        #         plt.figure(figsize=(128,128),dpi=1)
        #         plt.plot(gen[i][0][0:30],gen[i][0][30:60],color='blue',linewidth=300)
        #         plt.plot(gen[i][1][0:30],gen[i][1][30:60],color='red',linewidth=300)
        #         plt.plot(gen[i][2][0:30],gen[i][2][30:60],color='green',linewidth=300)
        #         plt.axis('off')
        #         plt.xlim(0.,1.)
        #         plt.ylim(0.,1.)
        #         plt.xticks(np.arange(0,1,0.1))
        #         plt.yticks(np.arange(0,1,0.1))
        #         if not os.path.exists("keras_gen"):
        #             os.makedirs("keras_gen")
        #         plt.savefig("keras_gen"+os.sep+str(i)+'.jpg',dpi=1)
        #         plt.close()
        # ##################################################################
        # #重整图片到0-1
        # else:
        #     for i in range(len(gen)):
        #         plt.plot(gen[i][0][0:30],gen[i][0][30:60],color='blue')
        #         plt.plot(gen[i][1][0:30],gen[i][1][30:60],color='red')
        #         plt.plot(gen[i][2][0:30],gen[i][2][30:60],color='green')
        #         plt.xlim(0.,1.)
        #         plt.ylim(0.,1.)
        #         plt.xticks(np.arange(0,1,0.1))
        #         plt.xticks(np.arange(0,1,0.1))
        #         plt.show()

