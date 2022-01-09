#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import numpy as np


class GradientDescent():
    """Descent gradient class with regularize technique
    
    Parameters
    ----------
    regularize : bool
        If True, the regularization is used.
    bias : bool
        If the True, a bias is added to the features.
    alpha : float > 0
        Coefficient for the step when updating the parameters.
    
    Notes
    -----
    This class aims at computing the parameters of a linear model using
    a descent gradient method with or without regularization.
    """
    
    def __init__(self, regularize=True, bias=True, alpha=3e-9):
        self.bias = bias
        if alpha < 0:
            raise ValueError('Alpha parameter must be > 0. Here {}.'.format(alpha))
        self.alpha = alpha
        self.regularize = regularize
        
        # set the epsilon value depending on the regularize case
        if regularize:
            self.epsilon = 1e-10
        else:
            self.epsilon = 1e-5
    
    def predict(self, new_features):
        """Make predictions using the result of the gradient descent
        
        Parameters
        ----------
        new_features : 2d sequence of float
            The feature for which to predict the labels.
            
        Returns
        -------
        predicted_labels : 2d sequence of float
            The predicted labels
        
        Notes
        -----
        The method fit must be called first.
        """
        
        if self.bias:
            new_features = self._add_bias(new_features)
        return self.hypothesis(new_features, self.parameters_)
    
    
    def fit(self, features, label, parameters=None):
        """Find the optimal parameters
        
        Parameters
        ----------
        features : 2d sequence of float
            The input parameters.
        label : 2d sequence of float
            The output parameters
        parameters : 2d sequence of float
            The initial guess for the descent gradient.
        """
        # add bias or not
        if self.bias:
            features = self._add_bias(features)
        
        # if no initial parameters are given get some randomly
        if parameters is None:
            n = features.shape[1]
            parameters = np.random.rand(n,1)
    
        # compute the initial prediction
        predictions = self.hypothesis(features, parameters)
        
        # solve depending of the regularization or not
        if self.regularize:
            self.parameters_ = self._regularize_fit(features, label, parameters, predictions)
        else:
            self.parameters_ = self._classic_fit(features, label, parameters, predictions)
         
        
    def _classic_fit(self, features, label, parameters, predictions):
        """Find the optimal parameters with classical method
        """

        costFct = 0
        costFctEvol = []
        count = 0
        # On utilise une boucle while
        while self.testCostFct(predictions, label, costFct, self.epsilon):
            count += 1
            costFct = self.costFunction(predictions, label)
            grads = self.gradients(predictions, label, features)
            parameters = self.updateParameters(parameters, grads, self.alpha)
            predictions = self.hypothesis(features, parameters)
            if count % 1000 == 0:
                print('%3i : cost function = {}'.format(costFct) % count)
            costFctEvol.append(costFct)
        print("\nFinish: {} steps, cost function = {}".format(count, costFct))
        return parameters
    
    def _regularize_fit(self, features, label, parameters, predictions):
        """Find the optimal parameters with regularized method
        """

        m = features.shape[0]
        lmb = (m * 0.2) / self.alpha
        
        costFct = 0
        costFctEvol = []
        count = 0
        while self.testRegCostFct(predictions, label, lmb, parameters, costFct, self.epsilon):
            count += 1
            costFct = self.regCostFunction(predictions, label, lmb, parameters)
            grads = self.regGradients(predictions, label, features, lmb, parameters)
            parameters = self.updateParameters(parameters, grads, self.alpha)
            predictions = self.hypothesis(features, parameters)
            if count % 10 == 0:
                print('%3i : cost function = {}'.format(costFct) % count)
            costFctEvol.append(costFct)
        print("\nFinish: {} steps, cost function = {}".format(count, costFct))
        return parameters
    
    def _add_bias(self, features):
        """Add bias column (1 vector)
        """
        bias = np.ones(features.shape[0])
        return np.column_stack([features, bias])
        
    def hypothesis(self, x, theta):
        """Compute our hypothesis model (linear regression), use a fonction:
        """
        return np.dot(x, theta)
    
    def costFunction(self, yhat, y):
        """Fonction de coût
        """
        return np.square(yhat - y).sum() / (2*y.shape[0])
    
    def regCostFunction(self, yhat, y, lmb, theta):
        """Fonction de coût régularisée
        """
        return self.costFunction(yhat, y) + lmb/(2*y.shape[0]) * np.square(theta).sum()
    
    def gradients(self, yhat, y, x):
        """Dérivée de la fonction de coût == gradients
        """
        return (((yhat - y) * x).sum(axis=0) / x.shape[0]).reshape(x.shape[1],1)

    def regGradients(self, yhat, y, x, lmb, theta):
        """Dérivée de la fonction de coût regularisée
        """
        return (((yhat - y) * x).sum(axis=0) / x.shape[0]).reshape(x.shape[1],1) + lmb/x.shape[0]*theta
    
    def updateParameters(self, parameters, grads, alpha):
        """Gradient descent: mise à jour des paramètres
        """
        return parameters - alpha * grads

    def testCostFct(self, yhat, y, prevCostFct, epsilon):
        """ Fonction pour tester l'évolution de la fonction de coût: vrai = continuer la descente de gradient
        """
        return np.abs(self.costFunction(yhat, y) - prevCostFct) >= epsilon*prevCostFct
    
    def testRegCostFct(self, yhat, y, lmb, theta, prevCostFct, epsilon):
        """ Fonction pour tester l'évolution de la fonction de coût régularisée
            
            Returns
            -------
            test : bool
                vrai = continuer la descente de gradient
        """
        return np.abs(self.regCostFunction(yhat, y, lmb, theta) - prevCostFct) >= epsilon*prevCostFct