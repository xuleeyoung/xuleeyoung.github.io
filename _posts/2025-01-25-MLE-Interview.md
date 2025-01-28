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

#### 5. Optimization method?

- Gradient Descent: 
  - Batch GD: Use entire dataset to compute the gradient at every step. Stable convergence but high computation cost
  - SGD: Use a single data point to compute gradient. Fast at every step, but slow convergence, may cause oscillation
  - Mini-batch GD: Divide the whole data set into many batches and compute gradient for each batch.
- Momentum: Adds a momentum term which can be represented as the moving average of previous gradients. Speeds up convergence, reduce oscillation.
- AdaGrad: Adjust the learning rate based on the frequency of parameter updates, giving smaller learing rate for frequently update parameters.
- RMSProp: Avoid AdaGrad's diminishing learning rate problem.
- Adam: Combines Momentum and RMSProp.

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

It reduces the internal covariance shift, which refers to the change in distribution of layer inputs during training. 

#### 14. Why residual networks work?
