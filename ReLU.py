import Layer
import numpy as np


#ReLu наиболее часто используемая функция активации при глубоком обучении. Максимум от f(x) = max(0,X)
#Возвращает 0, если принимает отрицательный аргумент, в случае же положительного аргумента, функция возвращает само число.
class ReLU(Layer.layer):
    def __init__(self):
        pass

    # создаем массив, вычисляем максимум и возвращаем его
    def forward(self, input: np.ndarray):
        output = np.zeros((input.shape[0], input.shape[1]))
        for i in range(input.shape[0]):
            output[i] = np.maximum(input[i], 0)
        return output

    #производная от релу производная от x - 1, 0-0
    def backward(self, input: np.ndarray, grad_output: np.ndarray):
        relu_grad = input > 0
        grad_output = grad_output * relu_grad
        return grad_output


