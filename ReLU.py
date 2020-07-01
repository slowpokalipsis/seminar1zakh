import Layer
import numpy as np


#ReLu �������� ����� ������������ ������� ��������� ��� �������� ��������. �������� �� f(x) = max(0,X)
#���������� 0, ���� ��������� ������������� ��������, � ������ �� �������������� ���������, ������� ���������� ���� �����.
class ReLU(Layer.layer):
    def __init__(self):
        pass

    # ������� ������, ��������� �������� � ���������� ���
    def forward(self, input: np.ndarray):
        output = np.zeros((input.shape[0], input.shape[1]))
        for i in range(input.shape[0]):
            output[i] = np.maximum(input[i], 0)
        return output

    #����������� �� ���� ����������� �� x - 1, 0-0
    def backward(self, input: np.ndarray, grad_output: np.ndarray):
        relu_grad = input > 0
        grad_output = grad_output * relu_grad
        return grad_output


