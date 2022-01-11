#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import numpy as np
import random
import matplotlib.pyplot as plt


class GradientDescent():
    
    """Ce module permet d'implementer une descente de gradient.
    
    Parametres
    ----------
    regularize : valeur possible (False, rigde, lasso, elasticNet)
    False : effectue une descente de gradient classique sans regularisation.
    rigde : effectue une descente de gradient avec regularisation Rigde.
    lasso : effectue une descente de gradient avec regularisation Lasso.
    elasticNet : permet d'effectuer une descente de gradient avec regularisation ElasticNet.
    bias : bool
        Si True, ajoute le biais sur les features.
    learning_rate : float > 0
        contitue le pas lors de la mise a jour des parametres.
    normalize : booleen
        Effectuer une normalisation sur les donnnees.
    
    Notes
    -----
    Cette classe vise à calculer les paramètres d'un modèle linéaire en utilisant
    une méthode de descente de gradient avec ou sans régularisation..
    """
    
    def __init__(self, regularize=False, bias=True, learning_rate=3e-9, normalize = False):
        self.bias = bias
        if learning_rate < 0:
            raise ValueError('learning_rate parameter must be > 0. Here {}.'.format(learning_rate))
        self.learning_rate = learning_rate
        self.regularize = regularize
        self.lambd = 0
        self.alpha = 0
        self.normalize = normalize
        
        
        if self.regularize:    
            self.epsilon = 1e-10
            if self.regularize == 'ridge':
                self.alpha = 0
            elif self.regularize == 'lasso':
                self.alpha = 1
            elif self.regularize == 'elasticNet':
                self.alpha = 0.4
            else :
                raise ValueError("le parametre 'regularize' ne peut prendre que : 'False', 'ridge', 'lasso', 'elasticNet'")
        else:
            self.epsilon = 1e-8
            self.lambd = 0
            self.alpha = 0
    
    def predict(self, new_features):
        """Faire des prédictions en utilisant le résultat de la descente de gradient.
        
        Paramètres
        ----------
        new_features : matrice de flottants.
            La caractéristique pour laquelle il faut prédire les étiquettes.
            
        Return
        -------
        predicted_labels : matrice de flottants.
            les predictions du modele
        
        Notes
        -----
        La méthode fit doit être appelée en premier.
        """
        
        if self.bias:
            new_features = self._add_bias(new_features)
        return self.hypothesis(new_features, self.parameters_)
    
    
    def fit(self, features, label, parameters=None, lambd=0, alpha=0):
        """Find the optimal parameters
        
        Parametres
        ----------
        features : matrice de flottants.
            Les donnees d'entrainements.
        label : matrice de flottants ou vecteur.
            Le etiquettes des donnees d'entrainements.
        parameters : matrice de flottants ou vecteur.
            Les parametres du modele.
        """
        
        if self.normalize:
            features = self.standardScaler(features)
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
        self.parameters_ = self._fit(features, label, parameters, predictions, self.lambd, self.alpha)

    
    def _fit(self, features, label, parameters, predictions, lambd, alpha):
        """Trouver les paramètres optimaux
        """

        m = features.shape[0]
   
        if self.regularize == 'rigde':
            self.lambd = lambd
            if self.alpha != 0 :
                raise ValueError("Le parametre 'alpha' ne concerne pas la regularisation Ridge")
            
        elif self.regularize == 'lasso':
            self.lambd = lambd
            if self.alpha != 1:
                raise ValueError("Le parametre 'alpha' ne concerne pas la regularisation Lasso")
                
        elif self.regularize == 'elasticNet':
            self.alpha = alpha
            self.lambd = lambd

    
        
        costFct = 0
        costFctEvol = []
        count = 0
        while self.testCostFct(predictions, label, self.lambd, self.alpha, parameters, costFct, self.epsilon):
            count += 1
            costFct = self.costFunction(predictions, label, self.lambd, self.alpha, parameters)
            grads = self.gradients(predictions, label, features, self.lambd, self.alpha, parameters)
            parameters = self.updateParameters(parameters, grads, self.learning_rate)
            predictions = self.hypothesis(features, parameters)
            costFctEvol.append(costFct)
            
        plt.xlabel('Iterations')
        plt.ylabel('Fonction cout')
        plt.title('Evolution de la fonction cout en fonctions des iterations')
        plt.plot(costFctEvol)
        return parameters
    
    def _add_bias(self, features):
        """Ajouter une colonne de bias (1 vecteur)
        """
        bias = np.ones(features.shape[0])
        return np.column_stack([features, bias])
        
    def hypothesis(self, x, theta):
        """Calculer notre modèle d'hypothèse (régression linéaire), utiliser une fonction :
        """
        return np.dot(x, theta)
    
    def _costFunction(self, yhat, y):
        """Fonction de coût
        """
        return np.square(yhat - y).sum() / (2*y.shape[0])
    def costFunction(self, yhat, y, lambd, alpha, theta):
        """Fonction de coût avec ou sans régularisée selon les parametres.
        """
        return self._costFunction(yhat, y) + lambd*(((1-alpha)/2) * np.square(theta).sum() + alpha*(np.abs(theta)).sum())
    


    def gradients(self, yhat, y, x, lambd ,alpha , theta):
        """Dérivée de la fonction de coût
        """

        return (((yhat - y) * x).sum(axis=0) / x.shape[0]).reshape(x.shape[1],1) + (lambd*(1-alpha))/x.shape[0]*theta
    
    def updateParameters(self, parameters, grads, learning_rate):
        """Gradient descent: mise à jour des paramètres
        """
        return parameters - learning_rate * grads
    
    def testCostFct(self, yhat, y, lambd, alpha, theta, prevCostFct, epsilon):
        """ Fonction pour tester l'évolution de la fonction de coût régularisée
            
            Returns
            -------
            test : bool
                vrai = continuer la descente de gradient
        """
        return np.abs(self.costFunction(yhat, y, lambd, alpha, theta) - prevCostFct) >= epsilon*prevCostFct
        
    
    def train_test_split(self, X, y, ratio=0.3, random_seed = 42):
        """ Fonction pour subdiviser les donnees en donnees d'entrainement et donnees de test.
        
             Parametres
            ----------
            X : matrice de flottants.
            Les donnees d'entrainements.
            
            y : matrice de flottants ou vecteur.
            Le etiquettes des donnees d'entrainements.
            
            ratio : La ratio du test set.
            
            Returns
            -------
            X_train : les features d'entrainement.
            y_train : les labels d'entrainement.
            X_test : les features de test.
            y_test : les labels de test.
            
        """
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        rows = len(X)
        random.seed(random_seed)
        test_index = random.sample(range(0,rows), int(rows*ratio))
        for i in range(rows):
            if i in test_index:
                X_test.append(X[i])
                y_test.append(y[i])
            else:
                X_train.append(X[i])
                y_train.append(y[i])
        return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
    
    def mae(self, y, y_pred) :
        """ Fonction determiner le score Mean Absolute Error.
        
             Parametres
            ----------
            y : matrice de flottants ou vecteur.
                Le etiquettes des donnees d'entrainements.
            y_pred : matrice de flottants ou vecteur.
                    valeurs predites par le modele.
            
            
            Returns
            -------
            score : flottant.
                le score par la metric Mean Absolute Error.
            
        """
        return np.abs(y_pred - y).sum() / (2*y.shape[0])
    
    def rmse(self, y, y_pred) :
        
        """ Fonction determiner le score Root Mean Squared Error.
        
             Parametres
            ----------
            y : matrice de flottants ou vecteur.
                Le etiquettes des donnees d'entrainements.
            y_pred : matrice de flottants ou vecteur.
                    valeurs predites par le modele.
            
            
            Returns
            -------
            score : flottant.
                le score par la metric Root Mean Squared Error.
            
        """
        return np.sqrt(np.square(y_pred - y).sum() / (2*y.shape[0]))
    
    def r2_score(self, y, y_pred) :
        """ Fonction determiner le score Coefficient de determination R2.
        
             Parametres
            ----------
            y : matrice de flottants ou vecteur.
                Le etiquettes des donnees d'entrainements.
            y_pred : matrice de flottants ou vecteur.
                    valeurs predites par le modele.
            
            
            Returns
            -------
            score : flottant.
                le score par la metric Coefficient de determination R2.
            
        """
        return 1 - (np.square(y_pred - y).sum() / np.square(y_pred - np.mean(y)).sum())
    
    
    def standardScaler(self, X):
        
        """ Fonction determiner le score Coefficient de determination R2.
        
             Parametres
            ----------
            X : matrice de flottants.
                les donnees d'entrainements a normaliser.
            
            Returns
            -------
             X : matrice de flottants.
                les donnees normalisees ( centrees reduites ).
            
        """
        return np.array((X - X.mean(axis=0)) / (X.std(axis=0)))
    
    def tunning(self,X,y,scoring,nb_params=10, validation_ratio = 0.3):
        """Ce module determine les hyperparametres optimale de regularisations 
    
        Parametres
        ----------
            X : matrice de flottants.
                les features des donnees d'entrainements.
            y: vecteur de flottants.
                les features des donnees d'entrainements.
            scoring : Le type de scoring pour evaluer les hyperparametres
                valeurs possible ('mae', 'rmse', 'r2_score')
            nb_params : Entier
                le nombre de hyperparamtres a tester
            validation_ratio : Flottant
                le ratio du validation set.
    
        """
        if self.regularize == False:    
            raise ValueError("Les hyperparametre de penalite s'applique uniquement sur une descent de gradient regularisee")
        else:
            best_param = 0
            if scoring not in ['rmse','mae','r2_score']:
                   raise ValueError("valeurs possible pour le scoring : 'mae', 'rmse', 'r2_score' ")
            a_tester = np.arange(0.01,1,round(1/nb_params, 2))
            X_train, y_train, X_validate, y_validate = self.train_test_split(X,y,ratio=validation_ratio)
        
            if self.regularize == 'ridge' or self.regularize == 'lasso':
               
                
                if scoring == "rmse":
                    best_score = 9e100
                    best_param = 0
    
                    for i in a_tester:
                        self.fit(X_train,y_train, lambd = i)
                        y_pred = self.predict(X_validate)

                        score = self.rmse(y_pred, y_validate)
                        if score < best_score:
                            best_score = score
                            best_param = i
                            
    
                    
                elif scoring == "mae":
                    best_score = 9e100
                    best_param = 0
    
                    for i in a_tester:
                        self.fit(X_train,y_train, lambd = i)
                        y_pred = self.predict(X_validate)
                        score = self.mae(y_pred, y_validate)
                        if score < best_score:
                            best_score = score
                            best_param = i
                    
                    
                elif scoring == "r2_score":
                    best_score = 1e-100
                    best_param = 0
    
                    for i in a_tester:
                        self.fit(X_train,y_train, lambd = i)
                        y_pred = self.predict(X_validate)
                        score = self.r2_score(y_validate, y_pred)
                        if score > best_score:
                            best_score = score
                            best_param = i
                            
                print("Le meilleur parametre lambda : ",best_param)
                    
            elif self.regularize == 'elasticNet':
                
                if scoring == "rmse":
                    best_score = 9e100
                    best_param = 0
    
                    for i in a_tester:
                        for j in a_tester:
                            self.fit(X_train,y_train, lambd = i, alpha = j)
                            y_pred = self.predict(X_validate)

                            score = self.rmse(y_pred, y_validate)
                            if score < best_score:
                                best_score = score
                                best_param = i , j
        
                    
                elif scoring == "mae":
                    best_score = 9e100
                    best_param = 0
    
                    for i in a_tester:
                        for j in a_tester:
                            self.fit(X_train,y_train, lambd = i, alpha = j)
                            y_pred = self.predict(X_validate)

                            score = self.mae(y_validate, y_pred)
                            if score < best_score:
                                best_score = score
                                best_param = i , j
                    
                elif scoring == "r2_score":
                    best_score = 0
                    best_param = 0 , 0
    
                    for i in a_tester:
                        for j in a_tester:
                            self.fit(X_train,y_train, lambd = i, alpha = j)
                            y_pred = self.predict(X_validate)
                            score = self.r2_score(y_pred, y_validate)
                            if score > best_score:
                                best_score = score
                                best_param = i , j
                print("Les meilleurs parametres de penalite pour la regularisation sont : lambda = ",best_param[0],", alpha = ",best_param[1])               