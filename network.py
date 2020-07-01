# -*- coding: utf8 -*-

import ReLU
import Dense
import Loss
import helper.mnist as mn
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from IPython.display import clear_output

# для чтения (чтоб получить shape используем mnist)

# структура сети
# входной полносвяз слой + релу + полносвязный слой + релу + слой
# dense(input, 100) -> relu() -> dense(100, 200) -> relu() -> dense(200, 10)
# последний слой выходной, он определяет картинки (есть ли на фото число 0-9)-> 10

"""
надо передать данные для фвд и бэк и придумать как эт обучать
механизм передачи данных между сетями
форвард - от нижнего к верхнему ()

fwd
сначала вход в инпут, потом поочередно входной знач в 0 слой, после из этого считывается
из этого результат выполнения слоя инпут и так пока не дойдет до последнего слоя
активайшонс будет хранить все результаты
акт -1 будет хранить выход из нейросети
"""

def forward(network, X):
    """
    Compute activations of all network layers by applying them sequentially.
    Return a list of activations for each layer.
    Make sure last activation corresponds to network logits.
    """
    activations = []
    input = X

    for i in range(len(network)):
        activations.append(network[i].forward(input))
        input = activations[-1]

    assert len(activations) == len(network)
    return activations

def predict(network, X):
    """
    Use network to predict the most likely class for each sample.
    """
    logits = forward(network, X)[-1]
    return logits.argmax(axis=-1)

# прин сеть, вход сигн и вых сигн
def train(network, X, y):
    """
    Train your network on a given batch of X and y.
    You first need to run forward to get all layer activations.
    You can estimate loss and loss_grad, obtaining dL / dy_pred
    Then you can run layer.backward going from last layer to first,
    propagating the gradient of input to previous layers.

    After you called backward for all layers, all Dense layers have already made one gradient step.
    """

    # Get the layer activations
    layer_activations = forward(network, X)
    # создаем новый список из вх д ко всем слоям - к лэйерс акт приб входной вектор
    layer_inputs = [X] + layer_activations  # layer_input[i] is an input for network[i]
    # рез вып-я нейросети
    logits = layer_activations[-1]

    # Compute the loss and the initial gradient софтмаксом
    loss = Loss.softmax_crossentropy_with_logits(logits, y)
    loss_grad = Loss.grad_softmax_crossentropy_with_logits(logits, y)

    # propagate gradients through network layers using .backward
    # hint: start from last layer and move to earlier layers
    # надо осуществить передачу от верхнего слоя к нижнему (бэк) - начнем с последнего
    for i in range(len(network) - 1, -1, -1):
        loss_grad = network[i].backward(layer_inputs[i], loss_grad)

    return np.mean(loss)

# дальше трейн луп с красивым выводом данных

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.random.permutation(len(inputs))
    for start_idx in trange(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

# х - вход, у - выход, они нужны для обучения и проверки

X_train, y_train, X_val, y_val, X_test, y_test = mn.load_dataset(flatten=True)

plt.figure(figsize=[6,6])
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.title("Label: %i"%y_train[i])
    plt.imshow(X_train[i].reshape([28,28]),cmap='gray');

input_shape = 0

network = []

network.append(Dense.Dense(X_train.shape[1], 100))
network.append(ReLU.ReLU())
network.append(Dense.Dense(100, 200))
network.append(ReLU.ReLU())
network.append(Dense.Dense(200, 10))

train_log = []
val_log = []

for epoch in range(25):

    for x_batch, y_batch in iterate_minibatches(X_train, y_train, batchsize=32, shuffle=True):
        train(network, x_batch, y_batch)

    train_log.append(np.mean(predict(network, X_train) == y_train))
    val_log.append(np.mean(predict(network, X_val) == y_val))
    test_log = np.mean(predict(network, X_test) == y_test)

    clear_output()
    print("Epoch", epoch)
    print("Train accuracy:", train_log[-1])
    print("Val accuracy:", val_log[-1])
    print("Test accuracy: ", test_log)
    plt.plot(train_log, label='train accuracy')
    plt.plot(val_log, label='val accuracy')
    plt.legend(loc='best')
    plt.grid()
    plt.show()