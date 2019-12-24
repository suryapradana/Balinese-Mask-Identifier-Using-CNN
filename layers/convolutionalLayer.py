import numpy as np
import math

def im2col(image, ksize, stride):
    # image is a 4d tensor([batchsize, width ,height, channel])
    image_col = []
    for i in range(0, image.shape[1] - ksize + 1, stride):
        for j in range(0, image.shape[2] - ksize + 1, stride):
            col = image[:, i:i + ksize, j:j + ksize, :].reshape([-1])
            image_col.append(col)
    
    image_col = np.array(image_col)
    
    return image_col


class ConvolutionalLayer(object):
    def __init__(self, shape, output_channels, ksize=3, stride=1, tipe_operasi = 'training', nama="none"):
        super(ConvolutionalLayer, self).__init__()
        self.nama = nama
        self.input_shape = shape
        self.output_channels = output_channels
        self.input_channels = shape[-1]
        self.stride = stride
        self.ksize = ksize
        
        if tipe_operasi == 'training':
            weights_scale = math.sqrt(ksize*ksize*self.input_channels/2)
            self.weights = np.random.standard_normal((ksize, ksize, self.input_channels, self.output_channels)) / weights_scale
            self.bias = np.random.standard_normal(self.output_channels) / weights_scale
        else:
            nama_file = 'temp/' + nama +'_lr0,001.npy'
            nama_file_bias = 'temp/' + nama +'_bias_lr0,001.npy'
            self.weights = np.load(nama_file)
            self.bias = np.load(nama_file_bias)
        
        self.eta = np.zeros((shape[0], int((shape[1] - ksize + 1) / self.stride), int((shape[2] - ksize + 1) / self.stride), self.output_channels))
        self.w_grad = np.zeros(self.weights.shape)
        self.b_grad = np.zeros(self.bias.shape)
        self.output_shape = self.eta.shape
              
        if (shape[1] - ksize) % stride != 0:
            print('input tensor height can not fit stride')
        if (shape[2] - ksize) % stride != 0:
            print('input tensor width can not fit stride')

    def simpan(self):
        nama_file = 'temp/' + self.nama +'_lr0,001.npy'
        nama_file_bias = 'temp/' + self.nama +'_bias_lr0,001.npy'
        np.save(nama_file, self.weights)
        np.save(nama_file_bias, self.bias)
    
    def forward(self, x, mode = 'training'):
        col_weights = self.weights.reshape([-1, self.output_channels])
        
        if mode == "testing":
            conv_out = np.zeros((1,self.eta.shape[1],self.eta.shape[2],self.eta.shape[3]))
            img_i = x
            self.col_image_i = im2col(img_i, self.ksize, self.stride)
            conv_out[0] = np.reshape(np.dot(self.col_image_i, col_weights) + self.bias, self.eta[0].shape)
        else:
            self.col_image = []
            conv_out = np.zeros(self.eta.shape)
            for i in range(x.shape[0]):
                img_i = x[i][np.newaxis, :]
                self.col_image_i = im2col(img_i, self.ksize, self.stride)
                conv_out[i] = np.reshape(np.dot(self.col_image_i, col_weights) + self.bias, self.eta[0].shape)
                self.col_image.append(self.col_image_i)
                
            self.col_image = np.array(self.col_image)
       
        return conv_out

    def gradient(self, eta):
        self.eta = eta

        col_eta = np.reshape(eta, [eta.shape[0], -1, self.output_channels])

        for i in range(eta.shape[0]):
            self.w_grad += np.dot(self.col_image[i].T, col_eta[i]).reshape(self.weights.shape)
        self.b_grad += np.sum(col_eta, axis=(0, 1))
        
        pad_eta = np.pad(self.eta, ((0, 0), (self.ksize - 1, self.ksize - 1), (self.ksize - 1, self.ksize - 1), (0, 0)), 'constant', constant_values=0)
        
        col_pad_eta = np.array([im2col(pad_eta[i][np.newaxis, :], self.ksize, self.stride) for i in range(eta.shape[0])])
        flip_weights = np.flipud(np.fliplr(self.weights))
        col_flip_weights = flip_weights.reshape([-1, self.input_channels])
        next_eta = np.dot(col_pad_eta, col_flip_weights)
        next_eta = np.reshape(next_eta, self.input_shape)
        
        return next_eta

    def backward(self, lr=0.00001):
        self.weights -= lr * self.w_grad
        self.bias -= lr * self.b_grad

        self.w_grad = np.zeros(self.w_grad.shape)
        self.b_grad = np.zeros(self.b_grad.shape)