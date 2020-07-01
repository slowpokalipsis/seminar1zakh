import numpy as np

class layer:
    def __init__(self):
        pass

    # значение с предыдущего уровня
    # batch is a part of epoch
    # epoch is когда весь датасет прошел через нейросеть в прямом и обратном направлении 1 раз
    # инпут  = массив [N на D]  N - размер бэтч, d - входной вектор в этот уровень
    # grad_output [N на D] N - колво бэтчей, D - вектор по количеству выходных связей
    def forward(self, input: np.ndarray):
        return input

    # обратное обучение зависит от инпута и от градиента выхода (значения спущенные с верхних уровней)
    # градиенты зависят от входа и от следующего (пред, мы идем обратно)  результата
    def backward(self, input: np.ndarray, grad_output: np.ndarray):
        # num_units = input.shape[1] # размерность
        return np.array([[], []]) #размерность выходна сигналов в пред уровне
