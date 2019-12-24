import numpy as np
from functools import reduce

class FullyConnectedLayer(object):
    def __init__(self, shape, output_num=2, tipe_operasi = 'training', nama="none"):
        self.input_shape = shape
        self.nama = nama

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
#        print('x ',x.shape)
        self.x = x.reshape([x.shape[0], -1])
#        print('self.x ',self.x.shape)
        output = np.dot(self.x, self.weights) + self.bias
        return output

    def gradient(self, eta):
        for i in range(eta.shape[0]):
            col_x = self.x[i][:, np.newaxis]
            eta_i = eta[i][:, np.newaxis].T
            self.w_grad += np.dot(col_x, eta_i)
            self.b_grad += eta_i.reshape(self.bias.shape)

        next_eta = np.dot(eta, self.weights.T)
        next_eta = np.reshape(next_eta, self.input_shape)
        return next_eta

    def backward(self, lr=0.00001):
        self.weights -= lr * self.w_grad
        self.bias -= lr * self.b_grad

        self.w_grad = np.zeros(self.w_grad.shape)
        self.b_grad = np.zeros(self.b_grad.shape)