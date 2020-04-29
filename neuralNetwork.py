from numba import jit, float32, int64, float64, typeof, int8
from numba.experimental import jitclass
import random
import numpy

@jit(nopython=True)
def flatt(arr, flatt_lenth):
    new = numpy.zeros((flatt_lenth))
    i = 0
    for x in arr:
        for y in x:
            new[i] = y
            i += 1
    return new

class Loss():

    @staticmethod
    @jit(nopython=True)
    def ErrorSqure(ans, y):
        return (ans - y[0]) ** 2

    @staticmethod
    @jit(nopython=True)
    def ErrorDoubleSqure(ans, y):
        return ans ** 2 - y[0] ** 2

    @staticmethod
    @jit(nopython=True)
    def Error(ans, y):
        return ans - y[0]

    @staticmethod
    @jit(nopython=True)
    def MSE(ans, y):
        x = 0
        for i in range(y.size):
            x += (ans - y[i]) ** 2
        if y.size == 0:
            raise Exception("y is empty")
        return x / y.size

class ActivationFunctions():

    @staticmethod
    @jit(nopython=True)
    def _sigmoid(x):
        if x > 40:
            return 1.0
        elif x < -12:
            return -10000
        else:
            return 1 / (1 + numpy.exp(-x))

    @staticmethod
    @jit(nopython=True)
    def sigmoid(x):
        l = len(x)
        arr = numpy.zeros((l))
        for y in range(l):
            arr[y] = ActivationFunctions._sigmoid(x[y])
        return arr


    @staticmethod
    @jit(nopython=True)
    def relu(x):
        arr = numpy.zeros((x.size))
        for y in range(x.size):
            arr[y] = x[y] * (x[y] > 0)
        return arr


    @staticmethod
    @jit(nopython=True)
    def softmax(x):
        return numpy.exp(x) / numpy.sum(numpy.exp(x), axis=0)



@jit(nopython=True)
def _m_t(b1, m_t, grad):
    return b1 * m_t + (1 - b1) * grad

@jit(nopython=True)
def _v_t(b2, v_t, grad):
    return b2 * v_t + (1 - b2) * (grad ** 2)

@jit(nopython=True)
def _m_t2(m_t, b1, t):
    return m_t / (1 - (b1 ** t))

@jit(nopython=True)
def _convert(convert, a, m_t, v_t, n):
    return convert - a * m_t / (numpy.sqrt(v_t) + n)


spec = [
    ("a", typeof(0.01)),
    ("b1", typeof(0.9)),
    ("b2", typeof(0.999)),
    ("n", typeof(10**-8)),
    ("t", typeof(1)),
    ("m_t", float32),
    ("v_t", float32)
]

class Optimizers():

    @jitclass(spec)
    class Adam():
        def __init__(self, a=0.01, b1=0.9, b2=0.999, n=10**-8):
            self.a = a
            self.b1 = b1
            self.b2 = b2
            self.n = n
            self.t = 0
            self.m_t = 0
            self.v_t = 0

        def optimize(self, convert, grad):
            copy = convert[:]
            while numpy.array_equal(convert, copy):
                self.t += 1
                # self.m_t = self.b1 * self.m_t + (1 - self.b1) * grad
                self.m_t = _m_t(self.b1, self.m_t, grad)
                # self.v_t = self.b2 * self.v_t + (1 - self.b2) * (grad ** 2)
                self.v_t = _v_t(self.b2, self.v_t, grad)
                # m_t = self.m_t / (1 - (self.b1 ** self.t))
                m_t = _m_t2(self.m_t, self.b1, self.t)
                # v_t = self.v_t / (1 - (self.b2 ** self.t))
                v_t = _m_t2(self.v_t, self.b2, self.t)
                # convert = convert - self.a * m_t / (numpy.sqrt(v_t) + self.n)
                convert = _convert(convert, self.a, m_t, v_t, self.n)
            return convert



class Layer():

    def __init__(self, n, func=lambda x: x):
        self.n = n
        self.func = func

    def _init(self, connect, optimizer, mw=1,mb=1, **kwargs):
        self.neurans = numpy.array([[random.random() for _ in range(connect)] for _ in range(self.n)])
        self.bias = random.random()
        self.biasOptimizer = optimizer(**kwargs)
        # self.optimizers = numpy.array([[optimizer(**kwargs) for _ in range(connect)] for _ in range(self.n)])
        self.optimizer = optimizer(**kwargs)
        del self.n

    @staticmethod
    @jit(nopython=True)
    def _predict(data, neurans, bias):
        r = numpy.zeros((neurans[0].size))
        for i in range(data.size):
            for n in range(r.size):
                r[n] = r[n] + neurans[i][n] * data[i]
        r = r + bias
        return r

    def predict(self, data):
        return self.func(Layer._predict(data, self.neurans, self.bias))
        # r = numpy.zeros((self.neurans[0].size))
        # for i in range(data.size):
        #     for n in range(r.size):
        #         r[n] = r[n] + self.neurans[i][n] * data[i]
        # r = r + self.bias
        # return self.func(r)

    @staticmethod
    @jit(nopython=True)
    def _optimize(optimizers, biasOptimizer, neurans, bias, loss):
        for i in range(optimizers.shape[0]):
            for x in range(optimizers[i].size):
                neurans[i][x] = optimizers[i][x].optimize(neurans[i][x], loss)
        bias = biasOptimizer.optimize(bias, loss)
        return optimizers, biasOptimizer, neurans, bias

    def optimize(self, loss):
        self.bias = self.biasOptimizer.optimize(numpy.array(self.bias), loss)[0]
        self.neurans = self.optimizer.optimize(self.neurans, loss)
        # for i in range(self.optimizers.shape[0]):
        #     for x in range(self.optimizers[i].size):
        #         self.neurans[i][x] = self.optimizers[i][x].optimize(self.neurans[i][x], loss)
        # self.bias = self.biasOptimizer.optimize(self.bias, loss)
        # self.optimizers, self.biasOptimizer, self.neurans, self.bias = Layer._optimize(self.optimizers, self.biasOptimizer, self.neurans, self.bias, loss)
        
        
class NeuralNetwork():

    def __init__(self, layers, optimizer, loss, **kwargs):
        self.layers = numpy.array(layers)
        self.loss = loss
        for l in range(self.layers.size):
            if l != self.layers.size - 1:
                self.layers[l]._init(layers[l+1].n, optimizer, **kwargs)
            else:
                self.layers[l]._init(0, optimizer, **kwargs)
    
    def predict(self, data):
        for layer in range(self.layers.size):
            if layer == self.layers.size - 1:
                return data
            data = self.layers[layer].predict(data)

    def train(self, data, answers, epochs=1):
        for _ in range(epochs):
            for i in range(data.size):
                print(i)
                p = self.predict(data[i])
                loss = self.loss(answers[i], p)
                for x in range(self.layers.size):
                    self.layers[x].optimize(loss)
