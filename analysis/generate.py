
import numpy as np
import copy
import scipy
from sklearn.covariance import GraphicalLassoCV, ledoit_wolf
import matplotlib.pyplot as plt

p = 100
n = 200


rm = np.random.random([100,100])
cov = rm@rm.T #已知的cov
mu = np.zeros(100) #已知的\mu
    
def get_data():
    x = np.random.multivariate_normal(mu, cov, (200)) #生成数据
    return x 

def eva(prec):
    #get the precision matrix
    # prec = np.linalg.inv(cov)#得到precision矩阵
    u,s,v = np.linalg.svd(prec)#对precision矩阵进行svd分解
    s_ = np.diag(np.sqrt(s))#对对角元素取开根号
    prec_half = u@s_@v #求得已知的precision矩阵的1/2 \theta^{1/2}

    #评估指标，表明得到的precision矩阵是否精确
    x_test = np.random.multivariate_normal(mu, cov, (80))#生成测试数据，用已经有的sample cov生成
    z_test = x_test@prec_half
    z_test_cov = z_test@z_test.T
    return z_test_cov

if __name__ == "__main__":
    
    #用graph lassso模型来估算数据的precision矩阵
    x = get_data()
    model = GraphicalLassoCV()
    model.fit(x)#用生成的数据来fit模型
    cov_ = model.covariance_ #得到估计的cov矩阵
    prec_ = model.precision_#得到估计的precision矩阵
    z_test_cov = eva(prec_)#得到估计的precision矩阵在test数据上的uncorrelated的表现
    
    plt.imshow(z_test_cov) #画出covariance矩阵

    #实验1：来测试不同的lambda 对估计的precision矩阵的影响
    # 画出graph lasso的alpha的最佳的值
    plt.figure(figsize=(4, 3))
    plt.axes([.2, .15, .75, .7])
    plt.plot(model.cv_results_["alphas"], model.cv_results_["mean_score"], 'o-')
    plt.axvline(model.alpha_, color='.5')
    plt.title('Model selection')
    plt.ylabel('Cross-validation score')
    plt.xlabel('alpha')
    plt.show()



# def prepare_data(self, x, num_samples, index):
#     m0 =np.mean(x, axis=0) 
#     # Estimation of mean and covariance
#     n, p = x.shape  #n = 500, p = 2048
#     x_ = x[num_samples:] #330, 2048 the last 330
#     m = np.mean(x_, axis=0) #2048, mean of every dimension
#     x_ = x_ - m #330, 2048
#     m = np.reshape(m, (1, p))# [1, 2048]
#     n, _ = x_.shape # n = 330
#     C = x_.T @ x_ / n# x^Tx/n
#     # Select training data
#     x = x[:num_samples] # 
#     m2 = np.mean(x, axis=0)
#     # Gaussian version
#     n, p = x.shape #170, 2048
    
#     mean = 0
#     std = 1
#     g = np.random.normal(mean, std, size=(n, p)).astype(np.float32)
#     # g_sampler = Sampler(g_args)
#     # g = g_sampler.sampling
    
#     # check the m0, m, m2 distribution
    
#     # file_name = '%s/%s_%s_%s_%s_%s_%s.png' % (self.dir, self.outstr1, self.outstr2, self.args.image_type, self.outstr3, index, 'gan_out_dist')
#     # if not os.path.isfile(file_name):
#     #     self.gan_output_dist(m0,m,m2,index)
#     # mean = mean_value(p, self.args.gmm['mean'])
#     # cov = cov_value(p, self.args.gmm['cov'])
#     # self.args.g_sample_method
#     # g = np.random.multivariate_normal(mean, cov, size=(n,))
#     # g = np.random.randn(n, p) #N(0,1)
#     if self.args.generate_gmm_data == True: #default is false
#         y = np.ones((n, 1))#[170,1]
#         g = y @ m + g @ scipy.linalg.sqrtm(C) #[170,1]@ [1,2048] + [170,2048]@C^1/2
#     return x, g

# def eigs(self, x, kernel=False):
#     u, s, v = np.linalg.svd(x)
#     return  s






# def cov_value(p, cov_value):
#     ### p is the multivariate number
#     cov_mtr = np.full((p,p),cov_value)
#     cov_mtr[np.diag_indices_from(cov_mtr)] = 1
#     return cov_mtr 

# def mean_value(p, mean_value):
#     m_vec = np.full((p,), mean_value)
#     return m_vec 

# mean_val = mean_value(p, 0)
# cov_val = cov_value(p, 1)
# values = np.random.multivariate_normal(mean_val, cov_val,(n,)).astype(np.float32)

# # g = np.random.multivariate_normal(mean, cov, size=(n,))