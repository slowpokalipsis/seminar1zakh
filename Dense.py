import numpy as np
import Layer

class Dense(Layer.layer):

    def __init__(self, input_units, output_units, learn_rate = 0.1): # ���-��, ���� ����-�
        self.learn_rate = learn_rate
        # 1d arr strok = kolvo vhod unit, stolb = vihod unit
        self.weights = np.random.randn(input_units, output_units) * 0.01
        self.bias = np.zeros(output_units) # ��������

    def forward(self, input: np.ndarray):
        #f(x) = input * W(weight) + B(��������) - ��������� � ������� ������� �������
        output = np.zeros((input.shape[0], len(self.bias)))
        for i in range(len(input)): # ����������� �� �������� �������� np.dot(Vect1, Vect2)
            temp = np.dot(input[i], self.weights) + self.bias
            output[i] = temp
        return output

    #df/dx = (df / d dense) * (d dense / dx), ����������������� .T ������� �����
    #������� ����������� ����� ��� ������ � ������� �� �� �� �������� � ����� ������������� ��������
    def backward(self, input: np.ndarray, grad_output: np.ndarray):
        t_weights = self.weights.T
        grad_input = np.dot(grad_output, t_weights)

        grad_weights = input.T.dot(grad_output) #����������� �����.� �� �������� �����������
        grad_bias = np.sum(grad_output, axis=0) #��������� �������� ����� (����� ����������� ��������)

        self.weights = self.weights - grad_weights * self.learn_rate #
        self.bias = self.bias - grad_bias * self.learn_rate

        return grad_input