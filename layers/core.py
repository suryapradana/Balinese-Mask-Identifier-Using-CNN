import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
from functools import reduce

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
            #msra method
            weights_scale = math.sqrt(ksize*ksize*self.input_channels/2)
            self.weights = np.random.standard_normal((ksize, ksize, self.input_channels, self.output_channels)) / weights_scale
            self.bias = np.random.standard_normal(self.output_channels) / weights_scale
        else:
            nama_file = 'temp/' + nama +'_lr0,001.npy'
            nama_file_bias = 'temp/' + nama +'_bias_lr0,001.npy'
            self.weights = np.load(nama_file)
            self.bias = np.load(nama_file_bias)
        
        # W - K + 2P + 1 / S
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
                # dot product inputan dengan filter(kernel) + bias, sebanyak jumlah filter(kernel)
                conv_out[i] = np.reshape(np.dot(self.col_image_i, col_weights) + self.bias, self.eta[0].shape)
                self.col_image.append(self.col_image_i)
                
            self.col_image = np.array(self.col_image)
       
        return conv_out

    def gradient(self, eta):
        self.eta = eta

        col_eta = np.reshape(eta, [eta.shape[0], -1, self.output_channels])
        #eta = (batchsize, size feature maps, size feature maps, channel deep size)
        for i in range(eta.shape[0]):
            # operasi dot product increment antara image dengan bobot
            self.w_grad += np.dot(self.col_image[i].T, col_eta[i]).reshape(self.weights.shape)
        self.b_grad += np.sum(col_eta, axis=(0, 1))
        
        pad_eta = np.pad(self.eta, ((0, 0), (self.ksize - 1, self.ksize - 1), (self.ksize - 1, self.ksize - 1), (0, 0)), 'constant', constant_values=0)
        
        col_pad_eta = np.array([im2col(pad_eta[i][np.newaxis, :], self.ksize, self.stride) for i in range(eta.shape[0])])
        # membalikkan bobot menjadi 180 derajat
        flip_weights = np.flipud(np.fliplr(self.weights))
        col_flip_weights = flip_weights.reshape([-1, self.input_channels])
        next_eta = np.dot(col_pad_eta, col_flip_weights)
        next_eta = np.reshape(next_eta, self.input_shape)
        
        return next_eta

    # update bobot
    def backward(self, lr=0.00001):
        #bobot = bobot - (learning rate * bobot gradient)
        self.weights -= lr * self.w_grad
        self.bias -= lr * self.b_grad

        self.w_grad = np.zeros(self.w_grad.shape)
        self.b_grad = np.zeros(self.b_grad.shape)



class MaxPoolingLayer(object):
    def __init__(self, shape, ksize=2, stride=2):
        self.input_shape = shape
        self.ksize = ksize
        self.stride = stride
        self.output_channels = shape[-1]
        self.index = np.zeros(shape)
        self.output_shape = [shape[0], shape[1] // self.stride, shape[2] // self.stride, self.output_channels]

    def forward(self, x):
        out = np.zeros([x.shape[0], x.shape[1] // self.stride, x.shape[2] // self.stride, self.output_channels])
        for b in range(x.shape[0]):
            for c in range(self.output_channels):
                for i in range(0, x.shape[1], self.stride):
                    for j in range(0, x.shape[2], self.stride):
                        #mencari nilai max pooling
                        out[b, i // self.stride, j // self.stride, c] = np.max(x[b, i:i + self.ksize, j:j + self.ksize, c])
                        #mencari index koordinat letak max pooling
                        index = np.argmax(x[b, i:i + self.ksize, j:j + self.ksize, c])
                        self.index[b, i + index // self.stride, j + index % self.stride, c] = 1
                        
        return out

    def gradient(self, eta):
        return np.repeat(np.repeat(eta, self.stride, axis=1), self.stride, axis=2) * self.index




class FullyConnectedLayer(object):
    def __init__(self, shape, output_num=2, tipe_operasi = 'training', nama="none"):
        self.input_shape = shape
        self.nama = nama
        print("fc: ", self.input_shape)

        input_len = int(reduce(lambda x, y: x * y, shape[1:]))
        
        if tipe_operasi == 'training':
            self.weights = np.random.standard_normal((input_len, output_num)) / 100
            self.bias = np.random.standard_normal(output_num) / 100
        else:
            nama_file = 'temp/' + nama +'_lr0,001.npy'
            nama_file_bias = 'temp/' + nama +'_bias_lr0,001.npy'
            self.weights = np.load(nama_file)
            self.bias = np.load(nama_file_bias)

        self.output_shape = [shape[0], output_num]

        self.w_grad = np.zeros(self.weights.shape)
        self.b_grad = np.zeros(self.bias.shape)
    
    def simpan(self):
        nama_file = 'temp/' + self.nama +'_lr0,001.npy'
        nama_file_bias = 'temp/' + self.nama +'_bias_lr0,001.npy'
        np.save(nama_file, self.weights)
        np.save(nama_file_bias, self.bias)

    def forward(self, x, mode="training"):
        print('x ',x.shape) #2, 22, 22, 24
        self.x = x.reshape([x.shape[0], -1]) #merubah menjadi vektor (2, 11.616)
        print('self.x ',self.x.shape)
        # dot product antara neuron input dengan bobot + bias
        output = np.dot(self.x, self.weights) + self.bias #output sebanyak 7 buah
        print("output forward fc : ", output)
        return output

    def gradient(self, eta):
        for i in range(eta.shape[0]):
            col_x = self.x[i][:, np.newaxis] #(11.616, 1)
            print("col_x : ", col_x.shape)
            eta_i = eta[i][:, np.newaxis].T #(1, 7)
            print("eta_i : ", eta_i.shape)
            self.w_grad += np.dot(col_x, eta_i) #(11.616, 7)
            print("w_grad fc : ", self.w_grad.shape)
            self.b_grad += eta_i.reshape(self.bias.shape) #(7)
            print("b_grad fc : ", self.b_grad.shape)

        # .T = Transpose, perkalian dot product
        next_eta = np.dot(eta, self.weights.T)
        next_eta = np.reshape(next_eta, self.input_shape)
        return next_eta

    # update bobot
    def backward(self, lr=0.00001):
        #bobot = bobot - (learning rate * bobot gradient)
        self.weights -= lr * self.w_grad
        self.bias -= lr * self.b_grad

        self.w_grad = np.zeros(self.w_grad.shape)
        self.b_grad = np.zeros(self.b_grad.shape)