from torchvision.models.densenet import densenet201
from analysis.args import *
from datagenerator.concgan_untrained import * 
from datagenerator.concgan_trained import *
import numpy as np 
from matplotlib import pyplot as plt
from analysis.analyzer import RMTanalyzer, group
import statsmodels.api as sm
import scipy.stats as stats
import sys 

from gan.simple_cgan import * 
from cnn.simple_cnn import * 
import logging  

from sys import platform
if platform == "linux" or platform == "linux2":
    opsys = 'linux'
elif platform == "darwin":
    opsys = 'mac'
elif platform == "win32":
    opsys = 'win'
else:
    opsys = 'unknown'

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='logs_file',
                    filemode='w')
# Until here logs only to file: 'logs_file'

# define a new Handler to log to console as well
console = logging.StreamHandler()
# optional, set the logging level
console.setLevel(logging.INFO)
# set a format which is the same for console use
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)


class RMT:
    def __init__(self, args):
        self.args = args 
        if args.gan_name != 'GAN':
            self.init_gan()
            self.init_cnn()

    def init_gan(self):
        
        # simple_gan = SimpleCGAN(self.args)
        # # simple_gan.train()
        # for i in range(200,300):
        #     simple_gan.exp_n = i
        #     # simple_gan.cnn.exp_n = i  
        #     simple_gan.generate_images()
    
        self.simple_gan = SimpleCGAN(args)
        # simple_gan.train()
        for i in range(0,500):
            self.simple_gan.exp_n = i
            # simple_gan.cnn.exp_n = i  
            self.simple_gan.generate_images() # list of num of classes elements [], each element is 200 smaple list
            
    def init_cnn(self):    
        self.cnn = SimpleCNN(args)
        # for i in range(0,500):
            


    @property
    def exp_n(self):
        return self._exp_n
    @exp_n.setter
    def exp_n(self, value):
        self._exp_n = value 
        
        
    @group('GAN', 'gan_name')
    def generate_cnn_rep(self): # this is for random matrix theory paper
        if self.args.image_type == 'GANuntrained':
            datagenerator = DataGeneratorUntrained(args)
        else:
            datagenerator = DataGenerator(args)
        imgname = self.args.z_sampler_args['name']
        file_name='images/{}/{}/{}'.format(self.args.image_type, imgname, self.args.classes[0])
        current_file_name = file_name + '_%d.png' % (self.args.num_imgs_per_class-1)
        
        if self.args.always_new == True:
            _, imgs = datagenerator.generate_images()
            data, _ = datagenerator.represent_images()
        else:
            if not os.path.isfile(current_file_name):
                _, imgs = datagenerator.generate_images()
            npy_file_name = 'data/%s_%s_%s_%s_%s.npy' % (imgname, self.args.image_type, self.args.classes[0], self.args.rep_model, self.args.truncation) 
            if not os.path.isfile(npy_file_name):
                data, _ = datagenerator.represent_images()
                p = data[0].shape[1]
            else:
                data = []
                for label in self.args.classes:
                    data.append(np.load('data/%s_%s_%s_%s_%s.npy' % (imgname, self.args.image_type, label, self.args.rep_model, self.args.truncation)))
        return data 

    @group('SimpleCGAN','gan_name')
    def generate_cnn_rep(self):  # this is for our method 
        
        npy_file_name = 'data/{}/{}/{}/{}_{}_{}.npy'.format(self.args.gan_name, self.args.run, self.exp_n, self.args.image_type, self.args.classes[0], self.args.rep_model)
        if not os.path.isfile(npy_file_name):
            
            data = self.cnn.represent_images()
            
            # #skipping save images to local, instead, directly save npy file to data directory
            # for i in self.args.classes: the other way, not using image,
            #     label_imges = img_samples[i]# recall the previous events
            #     image_array = self.cnn.represent_images(labeimg_samples, label) #feb-13 5:30  change to always save the image avary
        else:
            data = []
            for label in self.args.classes:
                d = np.load('data/{}/{}/{}/{}_{}_{}.npy'.format(self.args.gan_name, self.args.run, self.exp_n, self.args.image_type, label, self.args.rep_model))
                d = d.squeeze()
                data.append(d)
        return data 

    def rmtanalyzer_init(self,data):
        f = lambda x: np.tan(x)
        f = lambda x: x 
        data = [f(data[i]) for i in range(len(data))] #data, 3*500*2048, each class is 500 samples
        rmtanalyzer = RMTanalyzer(self.args, data)
        return rmtanalyzer
        

    @staticmethod
    def sqrtm(C):
        u, s, vh = np.linalg.svd(C)
        return u @ np.diag(np.sqrt(s)) @ vh

    def analysis(self,data):
        
        k = len(data)
        z = 1e-3
        C = {}
        m = {}
        for l in range(k):
            x = data[l]
            (n, p) = x.shape
            m[l] = np.mean(x, axis=0)
            C[l] = (x-m[l]).T @ (x-m[l]) / n
            
        deltas = 5*np.ones(k)
        S = 0
        for l in range(k):
                S = S + C[l]/(k * (1+deltas[l]))
        Q = np.linalg.inv(S + z*np.eye(p))

        for _ in range(30):
            S = 0
            for l in range(k):
                deltas[l] = np.trace(C[l] @ Q) / p
                S = S + C[l]/(k * (1+deltas[l]))
                
            Q = np.linalg.inv(S + z*np.eye(p))
            print(deltas)

        C = {}
        m = {}
        gdata = {}
        for l in range(len(data)):
            (n, p) = data[l].shape
            m[l] = np.mean(data[l], axis=0)
            C[l] = (data[l]-m[l]).T @ (data[l]-m[l]) / n
            g = np.random.randn(n, p)
            gdata[l] = m[l] + g @ RMT.sqrtm(C[l])

        x = np.concatenate((data[0], data[1], data[2]))
        (n, p) = x.shape
        G = x @ x.T / p
        lam, v = np.linalg.eig(G)

        g = np.concatenate((gdata[0], gdata[1], gdata[2]))
        (n, p) = g.shape
        gG = g @ g.T / p
        glam, gv = np.linalg.eig(gG)

        plt.scatter(v[:, 1], v[:, 2])
        plt.scatter(gv[:, 1], gv[:, 2], alpha=0.5)
        plt.show()


        print(self.args.rep_model)
        print(self.args.image_type)
        for i in range(k):
            print(m[i].T @ Q @ m[i] / p)

        deltas = []
        for z in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
            delta = 1
            for _ in range(10):
                Q = np.linalg.inv(C / (1+delta) + z*np.eye(p))
                delta = np.trace(C @ Q) / p
            deltas.append(delta)

        plt.plot(deltas)

        import pylab as plt
        plt.hist(lam ** 0.1)
        plt.show()

        x_i = x[1:]
        xi = x[0]
        z = 1e-3
        Q_i = np.linalg.inv(x_i.T.dot(x_i)/n + z*np.eye(p))
        (1/p) * xi.T @ (Q_i @ xi)

        delta = []
        for z in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
            Q_i = np.linalg.inv(x_i.T.dot(x_i)/n + z*np.eye(p))
            delta.append((1/p) * xi.T @ (Q_i @ xi))
        
        delta0 = delta

        plt.plot(delta, '.')
        plt.plot(delta0, '.r')

        
    def draw_eigvalue(self):
        data = self.generate_cnn_rep()
        rmtanalyzer = self.rmtanalyzer_init(data)
        ev = rmtanalyzer.eig_values('eigenvalue', data[0].shape[1])

        fig, ax = plt.subplots(1,1,figsize=(10, 7))
            
        # esd_mean = np.mean(existed_x, axis=0) 
        num_points = 20
        ax.scatter(range(num_points-1),ev[1:num_points], color='red')

        # axes[1].scatter(range(esd_mean.shape[0]-5),esd_mean[5:], color='blue')
        
     

        if opsys == 'linux':
            fig.text(0.04, 0.5, 'eigenvalue', va='center', rotation='vertical') 
        else:
            fig.text(0.04, 0.5, r'($\frac{%s}{p}$)'%('eigenvalue'), va='center', rotation='vertical')   
            fig.text(0.5, 0.0, '2 - 20 eigenvalue', va='center', rotation='horizontal')   
            

        file_name_head = 'figures/{}/{}/eig_distribution/{}/matrices_{}_{}_'.format(self.args.gan_name,self.args.run, self.args.image_type, self.args.rep_model,ev.shape[0])
        
        dir = '/'.join(file_name_head.split('/')[:-1])
        if not os.path.exists(dir):
            os.makedirs(dir)
        plt.savefig(file_name_head+'eig_dist.png', bbox_inches='tight')
        
        

        
        

    def rmtanalysis(self):
        # np.sqrt(np.log(200)/200)*100
        self.exp_n = 0
        data = self.generate_cnn_rep()
        rmtanalyzer = self.rmtanalyzer_init(data)

        p = rmtanalyzer.data[0].shape[1]

        file_name = '%s/%s_%s_%s_%s.png' % (rmtanalyzer.dir, rmtanalyzer.outstr1, rmtanalyzer.outstr2, self.args.image_type, rmtanalyzer.outstr3)
        if self.args.gramm_analysis == True and os.path.isfile(file_name) == False:
            self.rmtanalyzer.concent_vs_gauss_hist(title=r'{\bf %s} (p = %s)' % (self.args.rep_model, p), 
                                    disc=20, image_type=self.args.image_type)
            # Eigenvectors alignement
            self.rmtanalyzer.concent_vs_gauss_eigvecs(j=0, sign1=-1, sign2=1, 
                image_type=self.args.image_type)
        

        for decomposition_type in ['singularvalue','eigenvalue']:
            file_name = '%s/%s_%s_%s_%s_%s.png' % (self.rmtanalyzer.dir, self.rmtanalyzer.outstr1, self.rmtanalyzer.outstr2, self.args.image_type, self.rmtanalyzer.outstr3,decomposition_type)
            if self.args.singular_analysis == True and (os.path.isfile(file_name) == False or True):
                self.rmtanalyzer.concent_hist(decomposition_type, p)


    def eig_distribution(self): ## here we loop every exp_n, this analysis is to look at the distribution of cnn rep
        # we can use python parallel process which is
        evs = []
        range_num = 1

        eig_file_name = 'data/{}/{}/eig_distribution/e_{}_{}.npy'.format(self.args.gan_name,self.args.run, self.args.image_type, self.args.rep_model)
        if os.path.isfile(eig_file_name):
            existed_x = np.load(eig_file_name)
            start = existed_x.shape[0]
        else:
            dir = '/'.join(eig_file_name.split('/')[:-1])
            if not os.path.exists(dir):
                os.makedirs(dir)
            start = 0

        end = min(start+range_num,500)
        for i in range(start,end):
            self.cnn.exp_n = i
            
            self.exp_n = i
            data = self.generate_cnn_rep()
            
            rmtanalyzer = self.rmtanalyzer_init(data)
            ev = rmtanalyzer.eig_values('eigenvalue', data[0].shape[1])
            evs.append(ev)

        evs = np.asarray(evs) #500 * 500 * p dimension, the first 500 is because we have 500 sample size
        # the second 500 is because we have 10 class, every class we select 50    
        if os.path.isfile(eig_file_name):
            esd = np.concatenate((existed_x, evs), axis=0)
        else:
            esd = evs  

        

        # draw distribution of second and third egivalues 
        np.save(eig_file_name, esd)
        
        if opsys != 'linux':
            fig, ax = plt.subplots()
            check_list = [1,2,3,4,5]
            offsets = np.linspace(0,1,len(check_list))
            for j in [1,2,3,4,5]:
                ax.boxplot(esd[:,j], positions= [offsets[j-1]], widths=0.15, patch_artist=True, manage_ticks=False)
                # ax.set_xticks([1, 2, 3], ['mon', 'tue', 'wed'])
            
            ax.set(xticks=offsets, xticklabels=['{} th'.format(j+1) for j in check_list])
            
            # ax.set_xticklabels(['{} th'.format(j+1) for j in check_list], fontsize=18)
                # label='{} eigenvalue'.format(j)
            fig.text(0.5, -0.02, 'eigvalue', va='center', rotation='horizontal')
            plt.legend(fontsize=30)
            
            # file_name_head = '%s/%s_%s_%s_%s_' % (rmtanalyzer.dir, rmtanalyzer.outstr1, rmtanalyzer.outstr2, self.args.image_type, rmtanalyzer.outstr3)

            file_name_head = 'data/{}/{}/eig_distribution/matrices_{}_{}_{}_'.format(self.args.gan_name,self.args.run, self.args.image_type, self.args.rep_model,esd.shape[0])
        


            plt.savefig(file_name_head+'eig_dist.png', bbox_inches='tight')
    
            # self.draw_evalu(file_name_head, esd)

            # existed_x = existed_x[:154, :]
            # plot histograms
            disc =15
            existed_x = esd

            fig, ((ax1, ax2,ax3), (ax4, ax5,ax6)) = plt.subplots(2,3,sharex=True, sharey= False, figsize=(15, 7))
            axes = [ax1, ax2,ax3, ax4, ax5,ax6]

            for (i,ax) in enumerate(axes):
                sm.qqplot(existed_x[:,i], ax=ax, line='s') 
            plt.savefig(file_name_head+'eig_qqplot.png')


            # fig, ax = plt.subplots(1,1,sharex=True, sharey= False, figsize=(20, 7))
            
            fig, ((ax1, ax2,ax3), (ax4, ax5,ax6)) = plt.subplots(2,3, sharex=False, sharey= True, figsize=(15, 7))
            axes = [ax1, ax2,ax3, ax4, ax5,ax6]

            # for i, a in enumerate(esd):
            for (i,ax) in enumerate(axes):
                sns.distplot(existed_x[:,i], hist=True, kde=False,
                                bins=int(500 / disc),
                                # color=colors[i],
                                # label=labels[i],
                                ax=ax, 
                                norm_hist=True,
                                hist_kws={'edgecolor': 'black'})

            # fig, axes = plt.subplots(1,2,sharex=True, sharey= False, figsize=(20, 7))
            
            # esd_mean = np.mean(existed_x, axis=0) 

            # axes[0].scatter(range(esd_mean.shape[0]),esd_mean, color='red')
            # axes[1].scatter(range(esd_mean.shape[0]-5),esd_mean[5:], color='blue')
            
            # xlabels = [r'All', r'Without First 5']
            # for ax, col in zip(axes, xlabels):
            #     ax.set_xlabel(col, fontsize=18) # set the label for each ax

            if opsys == 'linux':
                fig.text(0.04, 0.5, 'eigenvalue', va='center', rotation='vertical') 
            else:
                fig.text(0.04, 0.5, r'($\frac{%s}{p}$)'%('eigenvalue'), va='center', rotation='vertical')   

            plt.savefig(file_name_head+'eig_value.png')


            # fig, ax = plt.subplots()
            # check_list = [0,1,2,3,4,5]
            # offsets = np.linspace(0,1,len(check_list))
            # for j in [0,1,2,3,4,5]:
            #     ax.boxplot(existed_x[:,j], positions= [offsets[j-1]], widths=0.15, patch_artist=True, manage_xticks=False)
            #     # ax.set_xticks([1, 2, 3], ['mon', 'tue', 'wed'])
            
            # ax.set(xticks=offsets, xticklabels=['{} th'.format(j+1) for j in check_list])
            
            # # ax.set_xticklabels(['{} th'.format(j+1) for j in check_list], fontsize=18)
            #     # label='{} eigenvalue'.format(j)
            # fig.text(0.5, -0.02, 'eigvalue', va='center', rotation='horizontal')
            # plt.legend(fontsize=30)
            # plt.savefig('box.png', bbox_inches='tight')

    
    
    
        



if __name__ == '__main__': 
    sys.stderr = open('errorlog.txt', 'w')
    parser = get_parser()       
    args = parser.parse_args()
    
    args.num_imgs_per_class = 50 #default is 500
    if args.num_imgs_per_class < args.batch_size:
        args.batch_size = args.num_imgs_per_class

    args.num_gram_imgs_per_class = 50 #default is 170
    args.truncation = 1.0
    args.z_sampler_args = {'name':'random_normal_sample', 'contents':{'mean':0.0, 'std':1.0}}
    # # args.g_sampler_args = {'name':'random_multinormal_sample', 'contents':{'mean':0.0, 'cov':0.0}}
    # # args.z_sampler_args = {'name':'bernoulli_sample', 'contents':{'v1':0, 'v2':1.0, 'pr':0.5}}
    # # args.z_sampler_args = {'name':'uniform_sample', 'contents':{'low':0, 'high':1}}
    args.g_sampler_args = {'name':'bernoulli_sample', 'contents':{'v1':-1, 'v2':1.0, 'pr':0.5}}
    # # args.g_sampler_args = {'name':'random_normal_sample', 'contents':{'mean':0.0, 'std':1.0}}
    
    args.classes = ['pizza', 'hamburger', 'mushroom']
    args.run = 'fully'
    args.gan_name = 'GAN'
    args.singular_analysis = True 
    args.sample_method = 'random_normal_sample'
    args.device='cpu'
    args.image_type = 'GANuntrained'
    args.always_new = True
    args.analysis_type = 'gan_ablation'
    args.pretrained = True

    
    for rep_model in ['resnet50']:
        args.rep_model = rep_model
        rmt = RMT(args)
        # rmt.generate_cnn_rep()
        # rmt.analysis()
        # rmt.rmtanalysis()
        rmt.draw_eigvalue()


    # args.z_sampler_args = {'name':'random_normal_sample', 'contents':{'mean':0.0, 'std':1.0}}
    # args.keras_gan = {'batch_size':100, 'epochs':400, 'sample_interval':10}
    # args.num_imgs_per_class = 200
    # # args.run = 'g_d_both_add_one_leak_relu'
    # args.run = 'fully_connected_as_cntk_suggested'
    # args.classes = [0,1,2,3,4,5,6,7,8,9]
    # args.represent_way = 'image'
    # args.num_gram_imgs_per_class = 50
    # args.sample_method = 'random_normal_sample'
    # args.singular_analysis = True 
    # args.gan_name = 'GAN' # SimpleCGAN or GAN
    # args.save_npy == 'False'
    # # args.rep_model_name = 'SimpleCNN'
    
    # for rep_model in ['SimpleCNN']:
    #     args.rep_model = rep_model
    #     args.image_type = 'GAN'
    #     args.device='cpu'
        
    #     rmt = RMT(args)
    #     # rmt.draw_evalu()
    #     #rmt.eig_distribution()
        
    #     # rmt.generate_cnn_rep()
    #     # rmt.rmtanalyzer_init()
    #     # rmt.analysis()
    #     rmt.rmtanalysis()
    sys.stderr.close()
    sys.stderr = sys.__stderr__





    

    
    
    
    
    
    
        



    