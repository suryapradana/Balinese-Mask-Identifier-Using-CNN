import numpy as np

class Relu(object):
    def __init__(self, shape):
        self.eta = np.zeros(shape)
        self.x = np.zeros(shape)
        self.output_shape = shape

    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    def gradient(self, eta):
        self.eta = eta
        self.eta[self.x < 0] = 0
        return self.eta



class Softmax(object):
    def __init__(self, shape):
        self.softmax = np.zeros(shape)
        self.eta = np.zeros(shape)
    
    def predict(self, prediction):
        exp_predict = np.zeros(prediction.shape)
        self.softmax = np.zeros(prediction.shape)

        for i in range(prediction.shape[0]):
            prediction[i, :] -= np.max(prediction[i, :])
            exp_predict[i] = np.exp(prediction[i])
            self.softmax[i] = exp_predict[i] / np.sum(exp_predict[i])
        return self.softmax

    def calc_loss(self, pred, label, mode="training"):
        self.label = label
        self.pred = pred
        self.predict(pred)
        self.loss = 0
        for i in range(pred.shape[0]):
            self.loss += np.log(np.sum(np.exp(pred[i]))) - pred[i, label[i]]
        return self.loss

    def gradient(self):
        self.eta = self.softmax.copy()
        for i in range(self.eta.shape[0]):
            self.eta[i, self.label[i]] -= 1
        return self.eta