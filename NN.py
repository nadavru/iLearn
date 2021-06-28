import numpy as np
from abc import abstractmethod
import math

class Module():

    @abstractmethod
    def forward(self, x):
        pass
    
    @abstractmethod
    def backward(self, loss):
        pass
        
    @abstractmethod
    def train(sel):
        pass
    
class Loss():

    @abstractmethod
    def forward(self, x, y):
        pass
    
    @abstractmethod
    def backward(self):
        pass

class Tanh(Module):

    def forward(self, x: np.array):
        self.temp = (np.exp(x) + np.exp(-x))
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    def backward(self, loss: np.array):
        return 4*loss/np.square(self.temp)
    
    def train(self):
        pass

class ReLU(Module):

    def forward(self, x: np.array):
        self.tmp = (x>0)
        return x * self.tmp
    
    def backward(self, loss: np.array):
        return self.tmp * loss
    
    def train(self):
        pass

class Sigmoid(Module):

    def forward(self, x: np.array):
        self.tmp = np.exp(x)
        return self.tmp /(self.tmp + 1)
    
    def backward(self, loss: np.array):
        return self.tmp / np.square(self.tmp + 1) * loss
    
    def train(self):
        pass

class Linear(Module):

    def __init__(self, in_dim: int, out_dim: int, with_b: bool = True, lr: float = 0.01) -> None:
        self.weights = np.random.randn(in_dim, out_dim)
        self.biases = np.random.randn(1, out_dim) if with_b else np.zeros((1, out_dim))
        self.with_b = with_b
        self.lr = lr

    def forward(self, x: np.array):
        self.x = x.copy()
        return x @ self.weights + self.biases
    
    def backward(self, loss: np.array):
        self.grad_w = self.x.transpose() @ loss
        self.grad_b = np.sum(loss, axis=0, keepdims=True)
        return loss @ self.weights.transpose()
        
    def train(self):
        self.weights -= self.lr*self.grad_w
        if self.with_b:
            self.biases -= self.lr*self.grad_b

class Sequential(Module):

    def __init__(self, hidden_dims: list, with_b: bool, activation, lr: float) -> None:
        layers = []
        input_dim = 2
        for dim in hidden_dims:
            layers.append(Linear(input_dim, dim, with_b, lr))
            if activation is not None:
                layers.append(activation())
            input_dim = dim
        layers.append(Linear(input_dim, 1, with_b, lr))
        self.layers = layers
    
    def forward(self, x: np.array):
        input = x
        for layer in self.layers:
            input = layer.forward(input)
        return input
    
    def backward(self, loss: np.array):
        input = loss
        for layer in reversed(self.layers):
            input = layer.backward(input)
        return input
        
    def train(self):
        for layer in self.layers:
            layer.train()

class MSELoss(Loss):

    def forward(self, x: np.array, y: np.array):
        self.x = x.copy()
        self.y = y.copy()
        return np.mean(np.square(x-y), axis=0)[0]
    
    def backward(self):
        return 2 * (self.x-self.y) / self.x.shape[0]

class Function():

    def __init__(self, string) -> None:
        if len([letter for letter in string.replace("pi","").replace("e","").replace("x","").replace("y","") if letter.isalpha()]):
            print("Abort.")
            exit()
        self.string = string.replace("^","**").replace("e",f"{math.e}").replace("pi",f"{math.pi}")

    def forward(self, x: np.array, y: np.array):
        return eval(self.string).reshape((-1,1))

class NN():

    def __init__(self, hidden_dims: list, with_b: bool, activation, lr: float, f_string: str, epochs: int, batch_size: int) -> None:
        num_of_epochs = 20
        self.model = Sequential(hidden_dims, with_b, activation, lr)
        self.criterion = MSELoss()
        self.F = Function(f_string)
        self.batch = np.random.rand(batch_size, 2) * 4 - 2
        self.test = np.random.rand(200, 2) * 4 - 2
        self.test = (self.test*1000).astype(int)/1000
        self.epochs = epochs
        self.indexes = [int(i*self.epochs/num_of_epochs)-1 for i in range(1,num_of_epochs+1)]\
            if num_of_epochs<=self.epochs else [i for i in range(self.epochs)]
        url = "https://i-learn-ml.oa.r.appspot.com/viewer/viewerGD.html#"
        self.disp_url = url+f"xmax=2.1&xmin=-2.1&ymax=2.1&ymin=-2.1&func={f_string}&points="
    
    def add_points(self):
        predict = self.model.forward(self.test)
        predict = (predict*1000).astype(int)/1000
        for i in range(predict.shape[0]):
            self.disp_url += f"({self.test[i,0]},{self.test[i,1]},{predict[i,0]})"
    
    def train(self):
        self.add_points()
        for i in range(self.epochs):
            predict = self.model.forward(self.batch)
            loss = self.criterion.forward(predict, self.F.forward(self.batch[:,0], self.batch[:,1]))
            self.model.backward(self.criterion.backward())
            self.model.train()
            if i in self.indexes:
                self.disp_url += "|"
                self.add_points()
        return self.disp_url
            

def runNN(
    hidden_dims="2,4,3", 
    activation="Tanh", 
    with_b=0,
    lr=0.1, 
    f_string="x*e^(-x^2-y^2)", 
    epochs=50000, 
    batch_size=500):

    hidden_dims = [int(dim) for dim in hidden_dims.split(",")]

    if activation.lower() == "tanh":
        activation = Tanh
    elif activation.lower() == "relu":
        activation = ReLU
    else:
        activation = Sigmoid

    nn = NN(
        hidden_dims=hidden_dims, 
        activation=activation, 
        with_b=with_b,
        lr=lr, 
        f_string=f_string, 
        epochs=epochs, 
        batch_size=batch_size)

    src = nn.train()
    return src
