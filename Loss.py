import numpy as np

# �������� - ��������� ������������� ������� ��� ������������ ������
# �-� ����������� ������ ����������� � ������ ��� �� ���� ��� ������ ���������� �����������
# ������� ������������ ������������ ������ 0 ��� 1
# ��� �-� ������������ ��� ����� ������������� ����� ������� �� ������ ����

# ���������� ������
# crossentropy ��������� ��������
def softmax_crossentropy_with_logits(logits, reference_answers):
    # loss = a(correct) + log(sum(exp(logits)))
    """Compute crossentropy from logits[batch,n_classes] and ids of correct answers"""
    tmp_logits = logits[np.arange(len(logits)), reference_answers]

    xentropy = -tmp_logits + np.log(np.sum(np.exp(logits), axis=-1))

    return xentropy

# grad ���������
def grad_softmax_crossentropy_with_logits(logits, reference_answers):
    """Compute crossentropy gradient from logits[batch,n_classes] and ids of correct answers"""
    ones_for_answers = np.zeros_like(logits)
    ones_for_answers[np.arange(len(logits)), reference_answers] = 1

    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)

    return (softmax - ones_for_answers) / logits.shape[0]