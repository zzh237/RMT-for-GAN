from analysis.args import *
# from pt_concgantools import * 
from gan.keras_gan import *
# from cnn.simple_cnn import * 

class SimpleCGAN(KerasGAN):
    def __init__(self,args):
        
        subname = type(self).__name__ 
        # cnn = SimpleCNN(args)
        # cnn = None 
        KerasGAN.__init__(self, args, subname)

        # 构建和编译判别器
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=self.optimizer,
            metrics=['accuracy'])

        # 构建生成器
        self.generator = self.build_generator()

        self.build_logic()
    
   

    def build_logic(self):
        # 生成器输入噪音，生成假的图片
        z = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([z, label])
 
        # 为了组合模型，只训练生成器
        self.discriminator.trainable = False
 
        # 判别器将生成的图像作为输入并确定有效性
        prob = self.discriminator([img,label])
 
        # The combined model  (stacked generator and discriminator)
        # 训练生成器骗过判别器
        self.combined = Model([z, label], prob)
        self.combined.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        if self.epochs_start > 1:
            self.generator.load_weights(os.path.join(self.model_path,"G_model.h5"),by_name=True)
            self.discriminator.load_weights(os.path.join(self.model_path,"D_model.h5"),by_name=True)

    def build_generator(self):
        model = Sequential()
        
        model.add(Dense(64, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        
        model.add(Dense(units=np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))
        
        model_input = multiply([noise, label_embedding])
        
        img = model(model_input)

        return Model([noise,label], img)

    def build_discriminator(self):#discriminator will take real and fake images to train on 
        model = Sequential()
        
        model.add(Dense(1024, input_dim = np.prod(self.img_shape)))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))



        model.add(Dense(units=1, activation='sigmoid'))
        model.summary()
        
        img =Input(shape=self.img_shape)

        label = Input(shape=(1,), dtype='int32')

        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))
        flat_img = Flatten()(img)
        model_input = multiply([flat_img, label_embedding])

        prob = model(model_input)

        return Model([img, label], prob)

    def train(self):
        X_train, y_train = self.train_data 
        y_train = y_train.reshape(-1,1)
        # Labels for fake and real images           
        label_fake = np.zeros(self.batch_size)
        label_real = np.ones(self.batch_size) 
        for i in range(self.epochs_start, self.epochs+1):

            for _ in tqdm(range(self.batch_size)):
                # Generate fake images from random noiset
                noise= np.random.normal(0,1, (self.batch_size, self.latent_dim))
                # Select a random batch of real images from MNIST
                idx = np.random.randint(0, X_train.shape[0], self.batch_size)
                real_images = X_train[idx]
                real_labels = y_train[idx]
                
                fake_images = self.generator.predict([noise,real_labels])   #this should be between -1, 1, generator will only take the noize 

                # 训练判别器，判别器希望真实图片，打上标签1，假的图片打上标签0
                # self.discriminator.trainable=True
                d_loss_real = self.discriminator.train_on_batch([real_images, real_labels], label_real)
                d_loss_fake = self.discriminator.train_on_batch([fake_images, real_labels], label_fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  训练生成器
                # ---------------------

                # Train the generator (to have the discriminator label samples as valid)
                sampled_labels = np.random.randint(0, len(self.classes), self.batch_size).reshape(-1,1)
                g_loss = self.combined.train_on_batch([noise, sampled_labels], label_real)

                # 打印loss值
                
            # Draw generated images every 10 epoches     
            if i == 1 or ((i>self.epochs_start) & (i % self.sample_interval == 0)):
                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (i, d_loss[0], 100*d_loss[1], g_loss))
                
                # os.path.join(self.model_path, 'training_images','training_images %d.png' %epoch)
                self.draw_images(file_name = 'images/{}/mnist/{}/training_images/{}'.format(self.subname, self.args.run, 'training_images %d.png' %i))
        
        self.save_GAN(i)

    def generate_batch_images(self, label):
        np.random.seed(self.exp_n)
        noise = np.random.normal(0,1,(self.batch_size, self.latent_dim))
        sampled_labels = np.full((self.batch_size, 1),label)
        gen = self.generator.predict([noise, sampled_labels])
        gen = self.norm_coef * gen + self.norm_coef
        return noise, gen,sampled_labels


if __name__ == '__main__': 
    
    parser = get_parser()       
    args = parser.parse_args()
    # args.z_sampler_args = {'name':'bernoulli_sample', 'contents':{'v1':0, 'v2':1.0, 'pr':0.5}}
    args.z_sampler_args = {'name':'random_normal_sample', 'contents':{'mean':0.0, 'std':1.0}}
    args.keras_gan = {'batch_size':100, 'epochs':400, 'sample_interval':10}
    args.num_imgs_per_class = 200
    # args.run = 'g_d_both_add_one_leak_relu'
    args.run = 'fully_connected_as_cntk_suggested'
    args.classes = [0,1,2,3,4,5,6,7,8,9]
    args.represent_way = 'image'
    
    simple_gan = SimpleCGAN(args)
    # simple_gan.train()
    for i in range(200,300):
        simple_gan.exp_n = i
        # simple_gan.cnn.exp_n = i  
        simple_gan.generate_images()







