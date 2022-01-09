
from pytorch_pretrained_biggan_self import *
import seaborn as sns
import numpy as np
import os 
import copy
import matplotlib.pyplot as plt 

group_store = {}
def group(v,pointer):
    def dec(f):
        name = f.__qualname__
        group_store[(name, v)] = f
        def method(self, *args, **kwargs):
            true_value = getattr(self.args, pointer)
            f = group_store[(name, true_value)]
            # return f(self, **self.paras)
            return f(self, *args, **kwargs)
        return method
    return dec

# class Decorator():
#     def __init__(self, param, pointer):
#         self.param = param 
#         self.pointer = pointer 
#         self.group_store = {} 
#     def __call__(self, f):
#         name = f.__qualname__
#         self.group_store[(name, self.param)] = f
#         def method(*args, **kwargs):
#             f = self.group_store[(name, self.pointer)]
#             # return f(self, **self.paras)
#             return f(*args, **kwargs)
#         return method

    
# class Logging:
#     def __init__(self, level):
#         self.level = level
#     def __call__(self, func):
#         def wrapper(*args, **kwargs):
#             if self.level == "warn":
#                 logging.warn("%s is running" % func.__name__)
#             elif self.level == "info":
#                 logging.info("%s is running" % func.__name__)
#             return func(*args)
#         return wrapper


class RMTanalyzer(object):
    def __init__(self, args, data):
        self.args = args 
        self.data = data
        
        
        self.prepare_name()
        
        # self.n = n #510 600 the final data size, should less than 1500
        self.k = len(self.args.classes) # 3
          
        if self.args.analysis_type == 'gan_ablation':
            self.x = np.concatenate(self.data, axis=0)

        else:
            self.x, self.g = self.prepare_data(self.data[0], num_samples=self.args.num_gram_imgs_per_class, index=0 )
            for i in range(self.k - 1):
                x_tmp, g_tmp = self.prepare_data(self.data[i + 1], num_samples=self.args.num_gram_imgs_per_class, index = i+1)
                self.x = np.concatenate((self.x, x_tmp), axis=0)
                self.g = np.concatenate((self.g, g_tmp), axis=0)
            
        # file_name = 'data/{}/{}/{}_{}_x.npy'.format(self.args.gan_name, self.args.run, self.args.image_type, self.args.rep_model_name)
        # np.save(file_name, self.x)
        # file_name = 'data/{}/{}/{}_{}_g.npy'.format(self.args.gan_name, self.args.run, self.args.image_type, self.args.rep_model_name)
        # np.save(file_name, self.g)

    
    
    @group('SimpleCGAN', 'gan_name')
    def prepare_name(self):
        # out = [self.args.z_sampler_args['name']]
        # for key, value in self.args.z_sampler_args['contents'].items():
        #     out.append(key)
        #     out.append(str(value))
        # self.outstr1 = '_'.join(out)
        self.outstr1 = '_'.join([self.args.gan_name, self.args.run])


        self.outstr2 = '_'.join([self.args.rep_model]) 

        # prepare output dir
        self.dir = os.path.join('figures',self.args.gan_name, self.args.run)
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        self.outstr3 = ''
        # self.outstr3 = '_'.join([self.args.rep_model, str(self.args.truncation)]) 

    @group('GAN', 'gan_name')
    def prepare_name(self):
        out = [self.args.z_sampler_args['name']]
        for key, value in self.args.z_sampler_args['contents'].items():
            out.append(key)
            out.append(str(value))
        self.outstr1 = '_'.join(out)


        out = [self.args.g_sampler_args['name']]
        g_args = copy.deepcopy(self.args.g_sampler_args)
        if 'n' not in g_args['contents']:
            g_args['contents']['n'] = self.args.num_gram_imgs_per_class
        if 'p' not in g_args['contents']:
            g_args['contents']['p'] = self.data[0].shape[1]
        
        for key, value in g_args['contents'].items():
            out.append(key)
            out.append(str(value))
        self.outstr2 = '_'.join(out) 

        # prepare output dir
        self.dir = os.path.join('figures',self.args.z_sampler_args['name'],self.args.g_sampler_args['name'])
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        self.outstr3 = '_'.join([self.args.rep_model, str(self.args.truncation)]) 


    def prepare_data(self, x, num_samples, index):
        m0 =np.mean(x, axis=0) 
        # Estimation of mean and covariance
        n, p = x.shape  #n = 500, p = 2048
        x_ = x[num_samples:] #330, 2048 the last 330
        m = np.mean(x_, axis=0) #2048, mean of every dimension
        x_ = x_ - m #330, 2048
        m = np.reshape(m, (1, p))# [1, 2048]
        n, _ = x_.shape # n = 330
        C = x_.T @ x_ / n# x^Tx/n

        # Select training data
        x = x[:num_samples] # 
        m2 = np.mean(x, axis=0)
        
        # Gaussian version
        n, p = x.shape #170, 2048
        g_args = copy.deepcopy(self.args.g_sampler_args)
        # g_args = self.args.g_sampler_args.copy()
        g_args['contents']['n'] = n
        g_args['contents']['p'] = p
        g_sampler = Sampler(g_args)
        g = g_sampler.sampling
        
        # check the m0, m, m2 distribution
       

        file_name = '%s/%s_%s_%s_%s_%s_%s.png' % (self.dir, self.outstr1, self.outstr2, self.args.image_type, self.outstr3, index, 'gan_out_dist')
        if not os.path.isfile(file_name):
            self.gan_output_dist(m0,m,m2,index)
        # mean = mean_value(p, self.args.gmm['mean'])
        # cov = cov_value(p, self.args.gmm['cov'])
        # self.args.g_sample_method
        # g = np.random.multivariate_normal(mean, cov, size=(n,))
        # g = np.random.randn(n, p) #N(0,1)
        if self.args.generate_gmm_data == True: #default is false
            y = np.ones((n, 1))#[170,1]
            g = y @ m + g @ sp.linalg.sqrtm(C) #[170,1]@ [1,2048] + [170,2048]@C^1/2
        return x, g

    def gan_output_dist(self, m0, m, m1,index):
        classname = self.args.classes[index]

        # f, ax = plt.subplots(figsize=(10, 7))
        f, axes = plt.subplots(1, 3, figsize=(21,5))
        ms = [m0,m,m1]
        for i in range(3):
            m_ = ms[i]
            sns.distplot(m_, hist = True, kde = True,
                    kde_kws = {'linewidth': 3}, ax=axes[i])
            # mod = mode(m_)
            # axes[0].axvline(mod, color='b', linestyle='-')
        
        # sns.distplot(m, hist = True, kde = True,
        #          kde_kws = {'linewidth': 3}, ax=axes[1])
        # sns.distplot(m1, hist = True, kde = True,
        #          kde_kws = {'linewidth': 3}, ax=axes[2])
        

        f.savefig('%s/%s_%s_%s_%s_%s_%s.png' % (self.dir, self.outstr1, self.outstr2, self.args.image_type, self.outstr3, classname, 'gan_out_dist'), bbox_inches='tight')
        # snsplt.figure.savefig('/Volumes/GoogleDrive/My Drive/rmt/figures/testdistribution/{}.png'.format(args['name']))
    

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

    def eigs(self, x, kernel=False):
        u, s, v = np.linalg.svd(x)
        return  s

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

    def eig_values(self, name, p):
        
        
        if name == 'singularvalue':
            esd_x = self.eigs(self.x)/p
            
        else:
            esd_x = np.square(self.eigs(self.x))/p

        return esd_x 

    def concent_hist(self, name, p):
        
        disc=20
        
        title=r'%s: {\bf %s} (p = %s)' % (name, self.args.rep_model, p)
        
        if name == 'singularvalue':
            esd_x = self.eigs(self.x)/p
            
        else:
            esd_x = np.square(self.eigs(self.x))/p

        labels = r'%s images' % self.args.image_type

        # plot histograms
        
        
        fig, axes = plt.subplots(1,2,sharex=True, sharey= False, figsize=(20, 7))
        
        axes[0].scatter(range(esd_x.shape[0]),esd_x, color='red')
        axes[1].scatter(range(esd_x.shape[0]-5),esd_x[5:], color='blue')
    
        xlabels = [r'All', r'Without First 5']
        for ax, col in zip(axes, xlabels):
            ax.set_xlabel(col, fontsize=18) # set the label for each ax

        fig.text(0.04, 0.5, r'($\frac{%s}{p}$)'%(name), va='center', rotation='vertical')
        
        # cols = ['all values', 'first 5 values']
        # for ax, col in zip(axes[0], cols):
        #     ax.set_title(col)
        
        # rows = ['value']
        # for ax, row in zip(axes[:,0], rows):
        #     ax.set_ylabel(row, rotation=0, size='large')
    
        # sns.distplot(esd_x, hist=True, kde=False,
        #                  bins=int(len(esd_x) / disc),
        #                  color='blue',
        #                  label=labels,
        #                  norm_hist=True,
        #                  hist_kws={'edgecolor': 'black'})

            
        # plt.legend(labels, fontsize=30)
        fig.suptitle(title, fontsize=40)
        # fig.suptitle('Various Straight Lines',fontweight ="bold")
        # plt.xlabel(r'Eigenvalues (ordered)', fontsize=30)
        # plt.ylabel(r'Value', fontsize=30)
        # ax.set_yscale('log')
        plt.savefig('%s/%s_%s_%s_%s_%s.png' % (self.dir, self.outstr1, self.outstr2, self.args.image_type, self.outstr3,name), bbox_inches='tight')
    

    def concent_vs_gauss_hist(self, title='', disc=10, image_type='GAN'):
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
        
        
            
        
        plt.savefig('%s/%s_%s_%s_%s.png' % (self.dir, self.outstr1, self.outstr2, image_type, self.outstr3), bbox_inches='tight')

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
            plt.savefig('%s/%s_%s_%s_%s_%s_%d.png' % (self.dir, self.outstr1, self.outstr2, image_type, self.outstr3, filenames[i], j), bbox_inches='tight')
