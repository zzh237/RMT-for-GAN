from analysis.args import *
# from pt_concgantools import * 
from gan.keras_gan import *
from PIL import Image
from analysis.analyzer import group



class SimpleCNN():
    def __init__(self,args):
        self.args = args 
        self.rep_model_name = 'SimpleCNN'
        self.image_type = 'GAN'
        self.gan_name = 'SimpleCGAN'
        self.classes = self.args.classes
        self.num_classes = len(self.classes)
        self.num_imgs_per_class = self.args.num_imgs_per_class

        self.train_data = self.load_data()
        x = self.train_data[0]
        shape = x.shape
        if x.ndim == 3:
            self.img_rows = shape[1]
            self.img_cols = shape[2]
            self.img_shape = (self.img_rows, self.img_cols, 1)
        
            self.channels = 1
            x = np.expand_dims(x, axis=3)
            self.train_data[0] = x 
        


        self.cnn = self.build_cnn()
        
        self.cnn.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adadelta(),
                    metrics=['accuracy'])

        if True:
            model_name = "model/{}/cnn.h5".format(self.rep_model_name)
            dir = '/'.join(model_name.split('/')[:-1])
            if not os.path.exists(dir):
                os.makedirs(dir)

            self.cnn.load_weights(model_name,by_name=False)


        @property
        def exp_n(self):
            return self._exp_n
        @exp_n.setter
        def exp_n(self, value):
            self._exp_n = value 
            
    def load_data(self):
        (x_train, y_train), (_, _) = mnist.load_data()

        x_train = x_train.astype('float32') #change from unit8 to float32
        # x_test = x_test.astype('float32')
        x_train /= 255
        # x_test /= 255
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        # y_test = keras.utils.to_categorical(y_test, self.num_classes)
        return [x_train, y_train]
    
    def build_cnn(self):

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                        activation='relu',
                        input_shape=self.img_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.summary()
        return model 

    
        # img = Input(shape=self.img_shape)
        # prob = model(img)
        
        # return Model(img, prob)
    
    @group('image', 'represent_way')
    def represent_images(self): 
        cnn_rep = []
        all_imgs = []
        print("Iteration %s, Extracting %s features of class: (%d)" % (self.exp_n, self.rep_model_name, len(self.classes)))
        for i, label in enumerate(self.classes):
            #print("Iteration %s, Extracting %s features of class: %s (%d/%d)" % (self.exp_n, self.rep_model_name, label, i + 1, len(self.classes)))
            if self.image_type == 'GAN':
                image_names = ['images/%s/mnist/%s/generated_images/%s/fake%s_%d.png' % (self.gan_name, self.args.run, self.exp_n, label, i) for i in range(self.num_imgs_per_class)]
            elif self.image_type == 'Real':
                image_names = glob.glob("imagenet/%s/*.JPEG" % label)
                image_names = image_names[:self.num_imgs_per_class]

            reps = []
            imgs = []
            for image_name in image_names:
                img = np.asarray(Image.open(image_name))
                x = img.astype('float32')
                x/=255 
                x = x.reshape(1,self.img_rows, self.img_cols, 1)
                out = self.get_output_function(4)(x)
                reps.append(out)

            cnn_rep.append(np.asarray(reps).squeeze()) #there are 10 number of 200*9260 matrices
        
            if self.args.save_npy == 'True':
                file_name = 'data/{}/{}/{}/{}_{}_{}.npy'.format(self.gan_name, self.args.run, self.exp_n, self.image_type, label, self.rep_model_name)
                dir = '/'.join(file_name.split('/')[:-1])
                if not os.path.exists(dir):
                    os.makedirs(dir)
                np.save(file_name, np.asarray(reps))
        return cnn_rep
    
    @group('noimage', 'represent_way')
    def represent_images(self, images, label): 
        reps = []
        for x in images:
            x/=255 
            x = x.reshape(1,self.img_rows, self.img_cols, 1)
            out = self.get_output_function(4)(x)
            reps.append(out)
        if self.args.save_npy == 'True':
            file_name = 'data/{}/{}/{}/{}_{}_{}.npy'.format(self.gan_name, self.args.run, self.exp_n, self.image_type, label, self.rep_model_name)
            dir = '/'.join(file_name.split('/')[:-1])
            if not os.path.exists(dir):
                os.makedirs(dir)
            np.save(file_name, np.asarray(reps))
            return np.asarray(reps) 
        else:
            return np.asarray(reps)#change list to array
        


    def get_output_function(self, output_layer_index):
        '''
        model: 要保存的模型
        output_layer_index：要获取的那一个层的索引
        '''
        vector_funcrion=K.function([self.cnn.layers[0].input],[self.cnn.layers[output_layer_index].output])
        def inner(input_data):
            vector=vector_funcrion([input_data])[0]
            return vector
        return inner
 


    
        
if __name__ == '__main__': 

    parser = get_parser()       
    args = parser.parse_args()
    # args.z_sampler_args = {'name':'bernoulli_sample', 'contents':{'v1':0, 'v2':1.0, 'pr':0.5}}
    args.z_sampler_args = {'name':'random_normal_sample', 'contents':{'mean':0.0, 'std':1.0}}
    args.keras_gan = {'batch_size':100, 'epochs':50, 'sample_interval':10}
    args.num_imgs_per_class = 200
    args.run = 'fully_connected_as_cntk_suggested'
    args.classes = [0,1,2,3,4,5,6,7,8,9]
    # model_raw, ds_fetcher, is_imagenet = selector.select('mnist', cuda = False )
    # print('zz')
    args.represent_way = 'image'
    cnn = SimpleCNN(args)
    for i in range(217,300):
        cnn.exp_n = i
        cnn.represent_images()