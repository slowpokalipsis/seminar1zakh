import numpy as np

# софтмакс - обобщение логистической функции дл€ многомерного случа€
# ф-€ преобразует вектор размерности в вектор той же разм где кажда€ координата полученного
# вектора представлена вещественным числом 0 или 1
# эта ф-€ примен€етс€ч дл€ задач классификации когда классов мб больше двух

# передаетс€ логиты
# crossentropy суммирует элементы
def softmax_crossentropy_with_logits(logits, reference_answers):
    # loss = a(correct) + log(sum(exp(logits)))
    """Compute crossentropy from logits[batch,n_classes] and ids of correct answers"""
    tmp_logits = logits[np.arange(len(logits)), reference_answers]

    xentropy = -tmp_logits + np.log(np.sum(np.exp(logits), axis=-1))

    return xentropy

# grad софтмакса
def grad_softmax_crossentropy_with_logits(logits, reference_answers):
    """Compute crossentropy gradient from logits[batch,n_classes] and ids of correct answers"""
    ones_for_answers = np.zeros_like(logits)
    ones_for_answers[np.arange(len(logits)), reference_answers] = 1

    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)

    return (softmax - ones_for_answers) / logits.shape[0]