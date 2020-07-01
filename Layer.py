import numpy as np

class layer:
    def __init__(self):
        pass

    # �������� � ����������� ������
    # batch is a part of epoch
    # epoch is ����� ���� ������� ������ ����� ��������� � ������ � �������� ����������� 1 ���
    # �����  = ������ [N �� D]  N - ������ ����, d - ������� ������ � ���� �������
    # grad_output [N �� D] N - ����� ������, D - ������ �� ���������� �������� ������
    def forward(self, input: np.ndarray):
        return input

    # �������� �������� ������� �� ������ � �� ��������� ������ (�������� ��������� � ������� �������)
    # ��������� ������� �� ����� � �� ���������� (����, �� ���� �������)  ����������
    def backward(self, input: np.ndarray, grad_output: np.ndarray):
        # num_units = input.shape[1] # �����������
        return np.array([[], []]) #����������� ������� �������� � ���� ������
