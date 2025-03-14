---
title: MLE Basic Interview QAs
date: 2025-01-25
categories: [Tech, Machine Learning]
tags: [interview, machine learning]     # TAG names should always be lowercase
description: Some basic MLE interview questions with short answers.
math: true
---

#### 1. What is supervised/unsupervised learning?

- The goal of supervised learning is to create a function that maps inputs to outputs. So the training data is always input features paired with output labels. Regression / Classification
- Unsupervised learning always trained with unlabeld data. Unsupervised learning tries to capture patterns or distribution of the input feature itself. Clustering / PCA

#### 2. What is discriminative/generative model?

- Discriminative model tries to directly learn the mapping from input features X to output labels Y. It directly models the conditional probability and use the learned mapping during inferrence.
- Generative model tries to learn the distribution of the input features itself. It models the joint probability $$P(X, Y)$$. During inferrence, you should use Bayes rule to compute $$P(Y \mid X)$$ and it can also generate new data samples.

#### 3. What is MLE/MAP?

- Maximum Likelihood Estimation is used to estimate the model paramaters by maximizing the likelihood of observed data $$P(D \mid w)$$ under the model. It focuses solely on the data and has no prior about the paramaters.
- Maximum a Posteriori maximize the probability $$P(w \mid D)$$. By Bayes rule, it is equivalent to $$P(D \mid w) P(w)$$. It combines the prior $$P(w)$$.

#### 4. What is overfitting? How to avoid?

Overfitting is a situation that the model fits too well on the training data, but performs poorly on new or unseen data. Symptoms: High accuracy on training set but low on test set.

1. Increase trianing set.
2. Simplify the model
3. Cross validation
4. Regularization
5. Drop-out/early stop/pruning
6. Ensemble method: Bagging/Boosting

**Dropout**: 训练时，以概率p随机丢弃神经元，等价于数据增强，类似于集成学习的bagging思想，每次随机选择部分特征，增强模型泛化能力。预测时，计算所有神经元，把神经元的权重乘p(训练时的输出期望)

#### 5. Optimization method?

- Gradient Descent: 
  - Batch GD: Use entire dataset to compute the gradient at every step. Stable convergence but high computation cost
  - SGD: Use a single data point to compute gradient. Fast at every step, but slow convergence, may cause oscillation
  - Mini-batch GD: Divide the whole data set into many batches and compute gradient for each batch.
- Momentum: Adds a momentum term which can be represented as the moving average of previous gradients. Speeds up convergence, reduce oscillation.
- AdaGrad: Adjust the learning rate based on the frequency of parameter updates, giving smaller learing rate for frequently update parameters.
- RMSProp: Avoid AdaGrad's diminishing learning rate problem.
- Adam: Combines Momentum and RMSProp.

```python
import numpy as np

# 目标函数和梯度
def f(x):
    return x**2 - 1  # 目标函数

def gradient(x):
    return 2 * x  # 梯度 f'(x)

# Adam 参数
learning_rate = 0.1  # 初始学习率
beta1 = 0.9  # 一阶动量衰减系数
beta2 = 0.999  # 二阶动量衰减系数
epsilon = 1e-8  # 防止除零
max_iterations = 100  # 最大迭代次数
tolerance = 1e-6  # 收敛阈值

# 初始化
x = np.random.randn()  # 随机初始化 x
m, v = 0, 0  # 初始化动量
t = 0  # 计数器

# Adam 迭代
for t in range(1, max_iterations + 1):
    grad = gradient(x)  # 计算梯度
    m = beta1 * m + (1 - beta1) * grad  # 更新一阶动量
    v = beta2 * v + (1 - beta2) * (grad ** 2)  # 更新二阶动量

    # 计算偏差修正
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    # Adam 更新
    x_new = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    # 收敛判断
    if abs(x_new - x) < tolerance:
        break

    x = x_new  # 更新 x

# 输出结果
print(f"最小值 x ≈ {x:.6f}, 对应 f(x) ≈ {f(x):.6f}")

```



```python
"""
Implements a two-layer Neural Network classifier in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import numpy as np
import random
import statistics

def fc_forward(x, w, b):
    """
    Computes the forward pass for a fully-connected layer.
    The input x has shape (N, d_in) and contains a minibatch of N
    examples, where each example x[i] has d_in element.
    Inputs:
    - x: A numpy array containing input data, of shape (N, d_in)
    - w: A numpy array of weights, of shape (d_in, d_out)
    - b: A numpy array of biases, of shape (d_out,)
    Returns a tuple of:
    - out: output, of shape (N, d_out)
    - cache: (x, w, b)
    """
    out = None
    out = x.dot(w) + b
    cache = (x, w, b)
    return out, cache


def fc_backward(dout, cache):
    """
    Computes the backward pass for a fully_connected layer.
    Inputs:
    - dout: Upstream derivative, of shape (N, d_out)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_in)
      - w: Weights, of shape (d_in, d_out)
      - b: Biases, of shape (d_out,)
    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d_in)
    - dw: Gradient with respect to w, of shape (d_in, d_out)
    - db: Gradient with respect to b, of shape (d_out,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    dw = x.T.dot(dout)
    db = np.sum(dout, axis=0)
    dx = dout.dot(w.T)

    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).
    Input:
    - x: Inputs, of any shape
    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    out = np.maximum(x, 0)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).
    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout
    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    dx = dout * (x > 0)

    return dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.
    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """

    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)

    loss = 0.0
    N = x.shape[0]
    dx = probs.copy()

    class_matrix = np.eye(x.shape[1], dtype=x.dtype)
    f = lambda y: class_matrix[y,:]
    Y_c = f(y).reshape((N, x.shape[1]))
    
    loss = -np.sum(Y_c * np.log(probs)) / N
    dx = (dx - Y_c)/ N

    return loss, dx


class TwoLayerNet:
    """
    A fully-connected neural network with softmax loss that uses a modular
    layer design.

    We assume an input dimension of D, a hidden dimension of H,
    and perform classification over C classes.
    The architecture should be fc - relu - fc - softmax.
    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.
    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=28*28, hidden_dim=100,
                 num_classes=10, weight_scale=1e-3):
        """
        Initialize a new network.
        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        """
        self.params = {}

        self.params["W1"] = np.random.normal(scale=weight_scale, size=(input_dim, hidden_dim))
        self.params["b1"] = 0.0
        self.params["W2"] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
        self.params["b2"] = 0.0

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.
        Inputs:
        - X: Array of input data of shape (N, d_in)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].
        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.
        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None

        hidden, cache1 = fc_forward(X, self.params["W1"], self.params["b1"])
        logits, cache2 = relu_forward(hidden)
        scores, cache3 = fc_forward(logits, self.params["W2"], self.params["b2"])

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}

        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Please do not reimplement softmax_loss, fc_backward, and #
        # relu_backward from scratch.                                              #
        ############################################################################
        loss, dout = softmax_loss(scores, y)
        dout, grads["W2"], grads["b2"] = fc_backward(dout, cache3)
        dout = relu_backward(dout, cache2)
        _, grads["W1"], grads["b1"] = fc_backward(dout, cache1)

        return loss, grads

```

```python
"""EECS545 HW2: Softmax Regression."""

import numpy as np
import math

def compute_softmax_probs(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Computes probabilities for logit x being each class.

    Inputs:
      - X: Numpy array of shape (num_data, num_features).
      - W: Numpy array of shape (num_class, num_features). The last row is a zero vector.
    Returns:
      - probs: Numpy array of shape (num_data, num_class). The softmax
        probability with respect to W.
    """
    probs = None

    logits = X.dot(W.T)
    logits -= np.max(logits, axis=1, keepdims=True)
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

    return probs


def gradient_ascent_train(X_train: np.ndarray,
                          Y_train: np.ndarray,
                          num_class: int,
                          max_iters: int = 300) -> np.ndarray:
    """Computes w from the train set (X_train, Y_train).
    This implementation uses gradient ascent algorithm derived from the previous question.

    Inputs:
      - X_train: Numpy array of shape (num_data, num_features).
                 Please consider this input as \\phi(x) (feature vector).
      - Y_train: Numpy array of shape (num_data, 1) that has class labels in
                 [1 .. num_class].
      - num_class: Number of class labels
      - max_iters: Maximum number of iterations
    Returns:
      - W: Numpy array of shape (num_class, num_features). The last row is a zero vector.
           We will use the trained weights on the test set to measure the performance.
    """
    N, d = X_train.shape  # the number of samples in training dataset, dimension of feature
    W = np.zeros((num_class, d), dtype=X_train.dtype)
    class_matrix = np.eye(num_class, dtype=W.dtype)

    int_Y_train = Y_train.astype(np.int32)
    alpha = 0.0005
    count_c = 0
    for epoch in range(max_iters):
        # A single iteration over all training examples
        delta_W = np.zeros((num_class, d), dtype=W.dtype)

        f = lambda y: class_matrix[y - 1,:]
        Y_c = f(int_Y_train).reshape((N, num_class))
        delta_W = (Y_c - compute_softmax_probs(X_train, W)).T.dot(X_train)

        W_new = W + alpha * delta_W
        W[:num_class-1, :] = W_new[:num_class-1, :]

        # Stopping criteria
        count_c += 1 if epoch > 300 and np.sum(abs(alpha * delta_W)) < 0.05 else 0
        if count_c > 5:
            break

    return W


def compute_accuracy(X_test: np.ndarray,
                     Y_test: np.ndarray,
                     W: np.ndarray,
                     num_class: int) -> float:
    """Computes the accuracy of trained weight W on the test set.

    Inputs:
      - X_test: Numpy array of shape (num_data, num_features).
      - Y_test: Numpy array of shape (num_data, 1) consisting of class labels
                in the range [1 .. num_class].
      - W: Numpy array of shape (num_class, num_features).
      - num_class: Number of class labels
    Returns:
      - accuracy: accuracy value in 0 ~ 1.
    """
    count_correct = 0
    N_test = Y_test.shape[0]
    int_Y_test = Y_test.astype(np.int32)

    Y_prob = compute_softmax_probs(X_test, W)
    Y_hat = np.argmax(Y_prob, axis=1) + 1
    count_correct = np.sum(Y_hat == Y_test.T)

    accuracy = count_correct / (N_test * 1.0)
    return accuracy

```



#### 6. Loss functions and their derivations?

- MSE: Regression. Gaussian noise model
- Binary Cross-Entropy: Classification. Bernoulli distribution
- Categorical Cross-Entropy: Multi-class classifiction. 
- Hinge Loss: SVM

#### 7. Bias/Variance

#### 8. Classifier evaluation metrics. Accuracy/Recall?

- accuracy: measures the percentage of correctly classified samples. Works well for balanced dataset.
- precision: of all observations that were predicted to be 1, what proportion were actually 1. Penalize false positives.
- recall: of all observations that were actually 1, what proportion did we predict to be 1. Penalize false negatives.
- F-1 score: harmonic mean of precision and recall.
- ROC-AUC: 

#### 9. Activation functions?

- Sigmoid
- tanh
- ReLU: computation efficiency, avoid vanishing gradients,  but dying neurons
- Leaky ReLU

#### 10. Data normalization and why?

Data normalization is a preprocessing technique. It is used to rescale the input values to ensure better convergence during backpropagation. If we don't do this then some of the features (those with high magnitude) will be weighted more in the cost function.

#### 11. Why convolution rather than FC layers?

1. Convolution preserves the spatial information from an image. FC layers have no relative spatial information.
2. Convolution reduce the complexity of the model, avoid the risk of overfitting.

#### 12. Why Max Pooling?

It is used to reduce the spatial dimension of the feature maps. It reduce the complexity of the model and avoid the risk of overfitting. 

#### 13. Batch Norm and why?

Batch normalization is a technique used to improve and accelerate the training in CNN.

It normalize the input for each layer over one batch to make training more stable and efficient.

It reduces the internal covariance shift, which refers to the change in distribution of layer inputs during training. 激活函数会改变各层数据的分布，并且随着网络加深影响越明显。

**加入缩放和平移变量的原因是：保证每一次数据经过归一化后还保留原有学习来的特征，同时又能完成归一化操作，加速训练。**

训练时：计算mini-batch的均值和方差，并维护一个指数移动平均值；

预测时：使用训练时计算的指数移动平均作为均值和方差，等价于在整个训练集上归一化。

#### 14. Why residual networks work?



