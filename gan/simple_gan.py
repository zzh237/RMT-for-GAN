from args import *
# from pt_concgantools import * 
from keras_gan import *

class SimpleGAN(KerasGAN):
    def __init__(self,args):
        
        subname = type(self).__name__ 

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
        img = self.generator(z)
 
        # 为了组合模型，只训练生成器
        self.discriminator.trainable = False
 
        # 判别器将生成的图像作为输入并确定有效性
        prob = self.discriminator(img)
 
        # The combined model  (stacked generator and discriminator)
        # 训练生成器骗过判别器
        self.combined = Model(z, prob)
        self.combined.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        if self.epochs_start > 1:
            self.generator.load_weights(os.path.join(self.model_path,"G_model.h5"),by_name=True)
            self.discriminator.load_weights(os.path.join(self.model_path,"D_model.h5"),by_name=True)


    def build_generator(self):
        model = keras.Sequential()
        
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
        img = model(noise)
        
        return Model(noise, img)

    def build_discriminator(self):#discriminator will take real and fake images to train on 
        model = keras.Sequential()
        
        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(1024))
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
        prob = model(img)

        return Model(img, prob)

    def build_generator2(self):
        model = keras.Sequential()
        
        model.add(Dense(units=256, input_dim=100))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Dense(units=512))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Dense(units=1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(units=784, activation='tanh'))
        model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
        return model
    def build_discriminator2(self):#discriminator will take real and fake images to train on 
        model = keras.Sequential()
        
        model.add(Dense(units=1024 ,input_dim=784))
        
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.3))
        # model.add(Dense(units=512))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.3))
        # model.add(Dense(units=256))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.3))
        
        model.add(Dense(units=1, activation='sigmoid'))
        
        model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
        return model

    def build_GAN(self, discriminator, generator):
        discriminator.trainable=False
        GAN_input = Input(shape=(100,))
        x = generator(GAN_input) # this value is between (-1,1), dim is (batch, 784)
        GAN_output= discriminator(x) # this value is between (0,1), dim is (batch, 1)
        GAN = keras.Model(inputs=GAN_input, outputs=GAN_output)
        GAN.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
        return GAN

    def generate_batch_images(self):
        noise = np.random.normal(0,1,(self.batch_size, self.latent_dim))
        gen = self.generator.predict(noise)
        gen = self.norm_coef * gen + self.norm_coef
        labels = None
        return noise, gen, labels

    

    def train(self):
        X_train, y_train = self.train_data 

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

                fake_images = self.generator.predict(noise)   #this should be between -1, 1, generator will only take the noize 

                
                
                # 训练判别器，判别器希望真实图片，打上标签1，假的图片打上标签0
                # self.discriminator.trainable=True
                d_loss_real = self.discriminator.train_on_batch(real_images, label_real)
                d_loss_fake = self.discriminator.train_on_batch(fake_images, label_fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  训练生成器
                # ---------------------

                # Train the generator (to have the discriminator label samples as valid)
                g_loss = self.combined.train_on_batch(noise, label_real)

                # 打印loss值
                
            # Draw generated images every 10 epoches     
            if i == 1 or ((i>self.epochs_start) & (i % self.sample_interval == 0)):
                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (i, d_loss[0], 100*d_loss[1], g_loss))
                
                # os.path.join(self.model_path, 'training_images','training_images %d.png' %epoch)
                self.draw_images(file_name = 'images/simple_gan/mnist/{}/{}/{}'.format(self.args.run,'training_images','training_images %d.png' %i))
        
        self.save_GAN(i)

    
    def train_GAN(self, epochs=1, batch_size=128):
    
        #Loading the data
        X_train, y_train = self.load_data()

        # Creating GAN
        generator= self.build_generator()
        discriminator= self.build_discriminator()
        GAN = self.build_GAN(discriminator, generator)

        for i in range(1, epochs+1):
            print("Epoch %d" %i)
            
            for _ in tqdm(range(batch_size)):
                # Generate fake images from random noiset
                noise= np.random.normal(0,1, (batch_size, 100))
                fake_images = generator.predict(noise)   #this should be between -1, 1, generator will only take the noize 

                # Select a random batch of real images from MNIST
                real_images = X_train[np.random.randint(0, X_train.shape[0], batch_size)]

                # Labels for fake and real images           
                label_fake = np.zeros(batch_size)
                label_real = np.ones(batch_size) 

                # Concatenate fake and real images 
                X = np.concatenate([fake_images, real_images])
                y = np.concatenate([label_fake, label_real])

                # Train the discriminator
                discriminator.trainable=True
                discriminator.train_on_batch(X, y)

                # Train the generator/chained GAN model (with frozen weights in discriminator) 
                discriminator.trainable=False
                GAN.train_on_batch(noise, label_real)

                # Draw generated images every 15 epoches     
            if i == 1 or i % 10 == 0:
                self.draw_images(generator, i)

        self.save_GAN(GAN)
    
    def discriminator(self):
        net = nn.Sequential(
            nn.Linear(784, 1),
            nn.Sigmoid()
        )
        return net

    
    
    
    def generator(self, noise_dim):
        net = nn.Sequential(
            nn.Linear(noise_dim, 784),
            nn.Tanh(),
        )
        return net

    def discriminator_loss(logits_real, logits_fake):   # 判别器的loss
        size = logits_real.shape[0]
        true_labels = torch.ones(size, 1).float()
        false_labels = torch.zeros(size, 1).float()
        bce_loss = nn.BCEWithLogitsLoss()
        loss = bce_loss(logits_real, true_labels) + bce_loss(logits_fake, false_labels)
        return loss
        
    



if __name__ == '__main__': 
    
    parser = get_parser()       
    args = parser.parse_args()
    # args.z_sampler_args = {'name':'bernoulli_sample', 'contents':{'v1':0, 'v2':1.0, 'pr':0.5}}
    args.z_sampler_args = {'name':'random_normal_sample', 'contents':{'mean':0.0, 'std':1.0}}
    args.simple_gan = {'batch_size':100, 'epochs':400, 'sample_interval':10}
    args.num_imgs_per_class = 500
    args.run = 'as_suggested_by_cntk'
    
    simple_gan = SimpleGAN(args)
    # simple_gan.train()
    simple_gan.generate_images()

    

