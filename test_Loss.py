import numpy as np

from Loss import softmax_crossentropy_with_logits, grad_softmax_crossentropy_with_logits
from helper.utils import eval_numerical_gradient

logits = np.linspace(-1, 1, 500).reshape([50, 10])
answers = np.arange(50) % 10

softmax_crossentropy_with_logits(logits, answers)
grads = grad_softmax_crossentropy_with_logits(logits, answers)
numeric_grads = eval_numerical_gradient(lambda l: softmax_crossentropy_with_logits(l, answers).mean(), logits)

assert np.allclose(numeric_grads, grads, rtol=1e-3,
                   atol=0), "The reference implementation has just failed. Someone has just changed the rules of math."
print("Well done!")
