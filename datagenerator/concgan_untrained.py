from typing import Optional, List
from pytorch_pretrained_biggan_self.utils import uniform_sample

import torch
import torch.nn as nn
from numpy.core._multiarray_umath import ndarray
from pytorch_pretrained_biggan_self import *
# from pytorch_pretrained_biggan_self import (BigGAN, one_hot_from_names, random_normal_sample,
#                                        truncated_noise_sample, binomial_sample, 
#                                        save_as_images)

import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import seaborn as sns
import numpy as np
import os 
import copy 
import scipy as sp
import glob
# from sklearn.preprocessing import normalize
from scipy.stats import mode 

sns.set(style="white", color_codes=True)
import matplotlib.pyplot as plt

sns.set_style('ticks')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
FONT_SIZE = 20
font = {'weight': 'bold',
        'size': FONT_SIZE}
import matplotlib

matplotlib.rc('xtick', labelsize=FONT_SIZE)
matplotlib.rc('ytick', labelsize=FONT_SIZE)
matplotlib.rc('font', **font)
sns.set_style('ticks')
matplotlib.rc('xtick', labelsize=18)
matplotlib.rc('ytick', labelsize=18)


class DataGeneratorUntrained(object):
    def __init__(self, args):
        self.args = args
        # ganmodel = BigGAN, num_imgs_per_class=500, classes=None, \
        # z_sampler_args={}, device='gpu',truncation=1
        if self.args.classes is None:
            self.classes = ['dog', 'cat', 'coffee']
        self.gan = BigGAGBaseline.from_pretrained('biggan-deep-256') #here we initialize the GAN model we are interested
        self.classes = self.args.classes
        self.num_imgs_per_class = self.args.num_imgs_per_class
        self.batch_size = self.args.batch_size
        self.sample_method = self.args.z_sampler_args['name']
        self.device = self.args.device
        self.z_sampler_args = self.args.z_sampler_args
        self.truncation = self.args.truncation 
        self.image_type= self.args.image_type

        self.rep_model_name=self.args.rep_model
        self.image_type=self.args.image_type
        self.pretrained=self.args.pretrained

        # image_type='GAN', 
        # rep_model_name='resnet18', 
        # pretrained=False,
        

    def generate_batch_images(self, batch_size=10, label='dog', device = 'gpu'):
        # Prepare an input
        # truncation = 1.0
        class_vector = one_hot_from_names([label] * batch_size, batch_size=batch_size)
        
        z_sampler_args = copy.deepcopy(self.z_sampler_args)
        z_sampler_args['contents']['n'] = batch_size 
        z_sampler_args['contents']['p'] = 128 

        z_sampler = Sampler(z_sampler_args)
        noise_vectors = z_sampler.sampling

        # noise_vectors = self.sample_dic[self.sample_method](batch_size = batch_size, truncation = truncation)
        # noise_vectors = truncated_noise_sample(truncation=truncation, batch_size=batch_size)
        noise_vector = noise_vectors
        # noise_vector = np.random.randn(batch_size, 128)

        # All in tensors
        noise_vector = torch.from_numpy(noise_vector)#10*128
        class_vector = torch.from_numpy(class_vector)#10*1000

        # Use GPU
        if self.device == 'gpu':
            noise_vector = noise_vector.to('cuda')
            class_vector = class_vector.to('cuda')
            self.gan.to('cuda')

        # Generate an image
        with torch.no_grad():
            images = self.gan(noise_vector, class_vector, self.truncation)

        # Go to CPU
        images = images.to('cpu')
        return noise_vectors, images

    def generate_images(self, save_imgs=True):
        all_noise_vectors = []
        all_images = []
        for i, label in enumerate(self.classes):
            print("Generating images of class: %s (%d/%d)" % (label, i + 1, len(self.classes)))
            noise_vectors, images = self.generate_batch_images(batch_size=self.batch_size, label=label)
            for i in range(self.num_imgs_per_class // self.batch_size - 1):
                noise_vec, imgs = self.generate_batch_images(batch_size=self.batch_size, label=label)
                images = torch.cat((images, imgs), 0)
                noise_vectors = np.concatenate((noise_vectors, noise_vec), axis=0)

            if save_imgs:
                save_as_images(images, file_name='images/{}/{}/{}'.format(self.image_type, self.sample_method, label))
            all_images.append(images)
            all_noise_vectors.append(noise_vectors)
        return all_noise_vectors, all_images

    def represent_images(self, save_representations=True):
        rep_model_name = self.rep_model_name
        image_type = self.image_type
        pretrained = self.pretrained

        cnn_rep = []
        all_imgs = []
        for i, label in enumerate(self.classes):
            print(
                "Extracting %s features of class: %s (%d/%d)" % (rep_model_name, label, i + 1, len(self.classes)))
            if image_type == 'GAN':
                image_names = ['images/%s/%s/%s_%d.png' % (self.image_type, self.sample_method, label, i) for i in range(self.num_imgs_per_class)]
            elif image_type == 'Real':
                image_names = glob.glob("imagenet/%s/*.JPEG" % label)
                image_names = image_names[:self.num_imgs_per_class]
            else:
                image_names = ['images/%s/%s/%s_%d.png' % (image_type, self.sample_method, label, i) for i in range(self.num_imgs_per_class)]

            if rep_model_name == 'resnet18':
                # Load the pretrained representation model
                rep_model = models.resnet18(pretrained=pretrained)
                # Use the model object to select the desired layer
                layer = rep_model._modules.get('avgpool')
            elif rep_model_name == 'resnet101':
                rep_model = models.resnet101(pretrained=pretrained)
                layer = rep_model._modules.get('avgpool')
            elif rep_model_name == 'resnet50':
                rep_model = models.resnet50(pretrained=pretrained)
                layer = rep_model._modules.get('avgpool')
            elif rep_model_name == 'densenet161':
                rep_model = models.densenet.densenet161(pretrained=pretrained)
                # Use the model object to select the desired layer
                new_classifier = nn.Sequential(*list(rep_model.classifier.children())[:-1])
                rep_model.classifier = new_classifier
            elif rep_model_name == 'densenet201':
                rep_model = models.densenet.densenet201(pretrained=pretrained)
                # Use the model object to select the desired layer
                new_classifier = nn.Sequential(*list(rep_model.classifier.children())[:-1])
                rep_model.classifier = new_classifier
            elif rep_model_name == 'vgg16':
                rep_model = models.vgg16(pretrained=pretrained)
                # Use the model object to select the desired layer
                new_classifier = nn.Sequential(*list(rep_model.classifier.children())[:-3])
                rep_model.classifier = new_classifier
            elif rep_model_name == 'vgg19':
                rep_model = models.vgg19(pretrained=pretrained)
                new_classifier = nn.Sequential(*list(rep_model.classifier.children())[:-3])
                rep_model.classifier = new_classifier
            elif rep_model_name == 'alexnet':
                rep_model = models.alexnet(pretrained=pretrained)
                new_classifier = nn.Sequential(*list(rep_model.classifier.children())[:-2])
                rep_model.classifier = new_classifier
            elif rep_model_name == 'googlenet':
                rep_model = models.inception_v3(pretrained=pretrained)
                new_classifier = nn.Sequential(*list(rep_model.fc.children())[:-1])
                rep_model.fc = new_classifier

            # Set representation model to evaluation mode
            rep_model.eval()

            # Normalization
            scaler = transforms.Resize((224, 224))
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            to_tensor = transforms.ToTensor()

            reps = []
            imgs = []
            for image_name in image_names:
                img = Image.open(image_name)
                # Create a PyTorch Variable with the transformed image
                t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
                tt_img = Variable(to_tensor(img))
                imgs.append(tt_img.data.numpy())

                # Create a vector of zeros that will hold our feature vector
                if rep_model_name == 'resnet18':
                    #    The 'avgpool' layer has an output size of 512
                    my_embedding = torch.zeros((1, 512, 1, 1))
                elif rep_model_name == 'resnet101':
                    my_embedding = torch.zeros((1, 2048, 1, 1))
                elif rep_model_name == 'resnet50':
                    my_embedding = torch.zeros((1, 2048, 1, 1))
                elif rep_model_name == 'vgg':
                    my_embedding = torch.zeros((1, 4096))
                elif rep_model_name == 'densenet161':
                    my_embedding = torch.zeros((1, 2208))
                elif rep_model_name == 'densenet201':
                    my_embedding = torch.zeros((1, 1920))
                elif rep_model_name == 'vgg16':
                    my_embedding = torch.zeros((1, 4096))
                elif rep_model_name == 'vgg19':
                    my_embedding = torch.zeros((1, 4096))
                elif rep_model_name == 'alexnet':
                    my_embedding = torch.zeros((1, 4096))
                elif rep_model_name == 'googlenet':
                    my_embedding = torch.zeros((1, 2048))

                def copy_data(m, i, o):
                    my_embedding.copy_(o.data)

                if rep_model_name[:6] == 'resnet':
                    # Attach that function to our selected layer
                    h = layer.register_forward_hook(copy_data)
                else:
                    h = rep_model.register_forward_hook(copy_data)

                # Run the model on our transformed image
                rep_model(t_img)

                # Detach our copy function from the layer
                h.remove()

                #  Return the feature vector
                if rep_model_name[:6] == 'resnet':
                    reps.append(my_embedding[0, :, 0, 0].data.numpy())
                else:
                    reps.append(my_embedding[0, :].data.numpy())

            cnn_rep.append(np.asarray(reps))
            if image_type =='GAN':
                all_imgs.append(np.asarray(imgs))
            if save_representations:
                np.save('data/%s_%s_%s_%s_%s.npy' % (self.sample_method, self.image_type, label, self.rep_model_name, str(self.truncation)), np.asarray(reps))

        return cnn_rep, all_imgs


