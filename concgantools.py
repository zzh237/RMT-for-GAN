from typing import Optional, List

import torch
import torch.nn as nn
from numpy.core._multiarray_umath import ndarray
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names,
                                       truncated_noise_sample,
                                       save_as_images)
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import seaborn as sns
import numpy as np
import scipy as sp
import glob
from sklearn.preprocessing import normalize

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


class DataGenerator(object):
    def __init__(self, num_imgs_per_class=500, classes=None):
        if classes is None:
            classes = ['dog', 'cat', 'coffee']
        self.gan = BigGAN.from_pretrained('biggan-deep-256')
        self.classes = classes
        self.num_imgs_per_class = num_imgs_per_class
        self.batch_size = 10

    def generate_batch_images(self, batch_size=10, label='dog'):
        # Prepare an input
        truncation = 1.0
        class_vector = one_hot_from_names([label] * batch_size, batch_size=batch_size)
        noise_vectors = truncated_noise_sample(truncation=truncation, batch_size=batch_size)
        noise_vector = noise_vectors
        # noise_vector = np.random.randn(batch_size, 128)

        # All in tensors
        noise_vector = torch.from_numpy(noise_vector)
        class_vector = torch.from_numpy(class_vector)

        # Use GPU
        noise_vector = noise_vector.to('cuda')
        class_vector = class_vector.to('cuda')
        self.gan.to('cuda')

        # Generate an image
        with torch.no_grad():
            images = self.gan(noise_vector, class_vector, truncation)

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
                save_as_images(images, file_name='images/%s' % label)
            all_images.append(images)
            all_noise_vectors.append(noise_vectors)
        return all_noise_vectors, all_images

    def represent_images(self, image_type='GAN', rep_model_name='resnet18', pretrained=True,
                         save_representations=True):
        cnn_rep = []
        all_imgs = []
        for i, label in enumerate(self.classes):
            print(
                "Extracting %s features of class: %s (%d/%d)" % (rep_model_name, label, i + 1, len(self.classes)))
            if image_type == 'GAN':
                image_names = ['images/%s_%d.png' % (label, i) for i in range(self.num_imgs_per_class)]
            elif image_type == 'Real':
                image_names = glob.glob("imagenet/%s/*.JPEG" % label)
                image_names = image_names[:self.num_imgs_per_class]

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
                np.save('data/%s_%s_%s.npy' % (image_type, label, rep_model_name), np.asarray(reps))

        return cnn_rep, all_imgs


class RMTanalyzer(object):
    def __init__(self, data, n=400, k=2):
        self.data = data
        self.n = n
        self.k = k
        self.x, self.g = self.prepare_data(self.data[0], num_samples=self.n // k)
        for i in range(k - 1):
            x_tmp, g_tmp = self.prepare_data(self.data[i + 1], num_samples=self.n // k)
            self.x = np.concatenate((self.x, x_tmp), axis=0)
            self.g = np.concatenate((self.g, g_tmp), axis=0)

    def prepare_data(self, x, num_samples):
        # Estimation of mean and covariance
        n, p = x.shape
        x_ = x[num_samples:]
        m = np.mean(x_, axis=0)
        x_ = x_ - m
        m = np.reshape(m, (1, p))
        n, _ = x_.shape # n here is 170 
        C = x_.T @ x_ / n

        # Select training data
        x = x[:num_samples] # number samples here is 170

        # Gaussian version
        n, p = x.shape
        
        
        g = np.random.randn(n, p)
        
        
        
        y = np.ones((n, 1))
        g = y @ m + g @ sp.linalg.sqrtm(C)
        return x, g

    

         

    def gram_eigs(self, x, kernel=False):
        # Data size and dimension
        n, p = x.shape

        # Gram matrix
        G = x @ x.T / p
        if kernel:
            k = lambda t: np.exp(-t)
        else:
            k = lambda t: t

        G = k(G)
        # Eigenvalues
        return np.linalg.eigvals(G) ** .1

    def gram_eigenvectors(self, x, kernel=False):
        # Data size and dimension
        n, p = x.shape

        # Gram matrix
        G = x @ x.T / p
        if kernel:
            k = lambda t: np.exp(-t)
        else:
            k = lambda t: t

        G = k(G)
        eigvals, eigvecs = np.linalg.eig(G)
        return eigvecs

    def concent_vs_gauss_hist(self, title='', filename='histogram', disc=10, image_type='GAN'):
        esd_x = self.gram_eigs(self.x)
        esd_g = self.gram_eigs(self.g)
        esd = [esd_x, esd_g]
        colors = ['black',
                  'green']
        labels = [r'%s images' % image_type,
                  r'Gaussian mixture']

        # plot histograms
        fig, ax = plt.subplots(figsize=(10, 7))
        for i, a in enumerate(esd):
            sns.distplot(a, hist=True, kde=False,
                         bins=int(len(a) / disc),
                         color=colors[i],
                         label=labels[i],
                         norm_hist=True,
                         hist_kws={'edgecolor': 'black'})

        plt.legend(labels, fontsize=30)
        plt.title(title, fontsize=40)
        plt.xlabel(r'Eigenvalues ($\lambda^{0.1}$)', fontsize=30)
        plt.ylabel(r'Density ($\log$ scale)', fontsize=30)
        ax.set_yscale('log')
        # plt.yticks([])
        # plt.xticks([])
        # plt.savefig('figures/%s.pdf' % filename, bbox_inches='tight')
        plt.savefig('figures/%s_%s.svg' % (image_type, filename), bbox_inches='tight')

    def concent_vs_gauss_eigvecs(self, j=0, sign1=0, sign2=0, filename='eigenvectors', image_type='GAN'):
        x_eigvecs = self.gram_eigenvectors(self.x)
        g_eigvecs = self.gram_eigenvectors(self.g)
        eigvecs = [x_eigvecs, g_eigvecs]
        colors = ['black',
                  'green']
        labels = ['Generated Images',
                  'Gaussian Mixture']
        filenames = ['concentr',
                     'gaussien']

        # plot histograms
        # fig, ax = plt.subplots()
        for i, a in enumerate(eigvecs):
            if i == 1:
                a[:, j] = sign1 * a[:, j]
                a[:, j + 1] = sign2 * a[:, j + 1]

            sns.jointplot(a[:, j], a[:, j + 1],
                          kind='kde',
                          space=0,
                          color=colors[i]).set_axis_labels(r"Eigenvector %d" % (j + 1), r"Eigenvector %d" % (j + 2),
                                                           fontsize=30).plot_joint(func=sns.kdeplot)
            plt.xticks([])
            plt.yticks([])
            plt.title(labels[i])
            # plt.savefig('figures/%s_%s_%d.pdf' % (filename, filenames[i], j), bbox_inches='tight')
            plt.savefig('figures/%s_%s_%s_%d.svg' % (image_type, filename, filenames[i], j), bbox_inches='tight')
