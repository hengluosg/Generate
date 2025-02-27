from scipy.stats import gaussian_kde
from sklearn.metrics.pairwise import rbf_kernel
import seaborn as sns
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import time
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

import pandas as pd
import statsmodels.formula.api as smf
from torch.utils.data import DataLoader, TensorDataset
# Offline Stage

from sklearn.linear_model import QuantileRegressor
from scipy.interpolate import interp1d

class QuantileRegressionNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
      
        super(QuantileRegressionNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            predictions = self.forward(x)
        return predictions

def quantile_loss(y_pred, y_true, tau):
    errors = y_true - y_pred
    return torch.mean(torch.max(tau * errors, (tau - 1) * errors))


def train_model_for_tau(X, Y, tau, hidden_dim=64, lr=0.001, epochs=100, batch_size=32):
   
    
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型和优化器
    model = QuantileRegressionNN(input_dim=X.shape[1], hidden_dim=hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练模型
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_Y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = quantile_loss(outputs, batch_Y, tau)
            loss.backward()
            optimizer.step()
    return model

class OfflineModel:
    def __init__(self, n, m, covariate_dim, theta_dim):
        self.n = n  # Number of covariate points
        self.m = m  # Number of quantiles
        self.covariate_dim = covariate_dim  # Dimension of covariates
        self.theta_dim = theta_dim  # Dimension of theta
        self.quantile_models = []



    #####linear model
    def train_multi_output_qrgmm(self,X,Y,m):   
        n, d = Y.shape
        quantile_grid = np.linspace(1/m, (m-1)/m, m-1)
        models = []

        for l in range(d):
            models_l = []
            for tau in quantile_grid:
                model = QuantileRegressor(quantile=tau, solver="highs")
                model.fit(X, Y[:, l])
                models_l.append(model)
            models.append(models_l)
        return models, quantile_grid


    ##### nn model
    # def train_multi_output_qrgmm(self, X, Y, m, hidden_dim=64, lr=0.001, epochs=200, batch_size=32):
    #     n, d = Y.shape
    #     quantile_grid = torch.linspace(1/m, (m-1)/m, m-1).numpy()

    #     all_models = []
    #     for l in range(d):
    #         models_for_l = []
    #         for tau in quantile_grid:
    #             model = train_model_for_tau(X, Y[:, l:l+1], tau, hidden_dim, lr, epochs, batch_size)
    #             models_for_l.append(model)
    #         all_models.append(models_for_l)

    #     return all_models, quantile_grid
    
    
    
    
class OnlineApplication:
    def __init__(self, quantile_models,quantile_points):
        
        self.quantile_models = quantile_models
        self.quantile_points = quantile_points
        
        



    def predicting_x0(self ,test_x, fit_models, quantile_points):
        
        
        n_test = test_x.shape[0]
        u = np.random.uniform(0, 1, size=n_test)
        interval = quantile_points[1] - quantile_points[0]
        ntau = len(quantile_points)

        low_indices = np.clip(np.floor(u / interval).astype(int), 0, ntau - 1)
        up_indices = np.clip(low_indices + 1, 0, ntau - 1)
        weights = (u - quantile_points[low_indices]) / interval

        gen_y = np.zeros(n_test)

        test_x_tensor = torch.tensor(test_x, dtype=torch.float32)
        # Predict all quantile values in a single step for efficiency
        quantile_predictions = np.array([model.predict(test_x_tensor[:, :covariate_dim]) for model in fit_models])

        for i in range(n_test):
            # Extract predictions for the current sample
            low_pred = quantile_predictions[low_indices[i], i]
            up_pred = quantile_predictions[up_indices[i], i]
            # Linearly interpolate to generate the sample
            gen_y[i] = low_pred + weights[i] * (up_pred - low_pred)

        return gen_y

    def predicting(self, test_x, fit_models, quantile_points):
   
        n_test = test_x.shape[0]
        u = np.random.uniform(0, 1, size=n_test)
        interval = quantile_points[1] - quantile_points[0]
        ntau = len(quantile_points)

        low_indices = np.clip(np.floor(u / interval).astype(int), 0, ntau - 1)
        up_indices = np.clip(low_indices + 1, 0, ntau - 1)
        weights = (u - quantile_points[low_indices]) / interval
        gen_y = np.zeros(n_test)
        test_x_tensor = torch.tensor(test_x, dtype=torch.float32)
        # Predict all quantile values in a single step for efficiency
        quantile_predictions = np.array([model.predict(test_x_tensor[:, :covariate_dim]).numpy() for model in fit_models])
        for i in range(n_test):
            # Extract predictions for the current sample
            low_pred = quantile_predictions[low_indices[i], i]
            up_pred = quantile_predictions[up_indices[i], i]
            # Linearly interpolate to generate the sample
            gen_y[i] = low_pred + weights[i] * (up_pred - low_pred)

        return gen_y



    def generate_solutions(self, x, M):
        input_dim = x.shape[0]
        output_dim = len(self.quantile_models)
        generated_data = np.zeros((M, output_dim))

        test_x = np.tile(x, (M, 1))  # Initial input is just x_star
        generated_data[:, 0] = self.predicting_x0(test_x, self.quantile_models[0], self.quantile_points)

        # Predict subsequent target variables
        for d in range(1, output_dim):
            test_x = np.hstack([np.tile(x, (M, 1)), generated_data[:, :d]])
            generated_data[:, d] = self.predicting(test_x, self.quantile_models[d], self.quantile_points)
        
        return generated_data
    
if __name__ == '__main__': 
    np.random.seed(42)  
    n = 100
    covariates = np.random.rand(n, 5)  # Example covariates
    thetas = np.random.rand(n, 1)  # Example thetas
    covariate_dim = covariates.shape[1]  # Dimension of covariates
    theta_dim = thetas.shape[1] # Dimension of theta
    m = 100  # Number of quantitles
    offline_model = OfflineModel(n, m, covariate_dim, theta_dim)
    models, quantile_grid = offline_model.train_multi_output_qrgmm(covariates, thetas, m)
    online_app = OnlineApplication(models, quantile_grid)
    x_new = np.random.rand(1, 5)  # New covariate value
    M  = 5000
    solutions = online_app.generate_solutions(x_new, M)
    print(solutions)
    













