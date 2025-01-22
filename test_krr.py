
from sklearn.model_selection import KFold
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

import statsmodels.formula.api as smf

# Offline Stage

from sklearn.linear_model import QuantileRegressor
from scipy.interpolate import interp1d
plt.rcParams['figure.max_open_warning'] = 150  # 设置警告阈值为 50




def grid_sample( d, lowbound, upbound, point):
        n = int(round(point ** (1.0 / d))) +1 
        grid_1d = np.linspace(lowbound + 0.1, upbound - 0.1, n)
        grid_points = np.array(list(product(grid_1d, repeat=d)))
        if len(grid_points) > point:
            indices = np.random.choice(len(grid_points), size=point, replace=False)
            grid_points = grid_points[indices]
        return grid_points




def project(theta, lowbound, upbound):

    return np.clip(theta, lowbound, upbound)

class OfflineModel:
    def __init__(self, n, m, K, covariate_dim, theta_dim):
        self.n = n  # Number of covariate points
        self.m = m  # Number of quantiles
        self.K = K  # Number of samples per covariate point
        self.covariate_dim = covariate_dim  # Dimension of covariates
        self.theta_dim = theta_dim  # Dimension of thet
        self.quantile_models = []

    #random search
    def first_stage(self, X, K):
        thetas = []
        objective_values = []
        best_theta_total = []
        for x in X:
            solutions = np.random.uniform(lowbound, upbound, size=(K, theta_dim)) 
            
            objectives = min([objective_function(theta, x) for theta in solutions])
            thetas.append(solutions)
            best_theta = solutions[np.argmin(objectives)]  
            best_theta_total.append(best_theta)
            objective_values.append(objectives)
        return   np.array(objective_values), np.array(best_theta_total)
    



    #SGD
    def first_stage2(self,X,  K):
        T = K
        thetas1 = []
        objective_values = []
        best_theta_total = []
        
        for x in X:
            theta = np.random.uniform(lowbound, upbound, theta_dim)   # Random initializations
            theta_av = []
            for t in range(T):
                
                grad = gradient(theta, x)  # Compute the gradient
                theta = theta - eta * grad  # Update theta using SGD
                theta = project(theta, lowbound=lowbound, upbound=upbound) 
                
                theta_av.append(theta)
                
            #best_theta_total.append(theta)
            solutions = theta_av
            objectives = [objective_function(theta, x) for theta in solutions]
            thetas1.append(solutions)
            best_theta = np.mean(np.array(theta_av),axis = 0)
            best_theta_total.append(best_theta)
            objective_values.append(min(objectives))
        return  np.array(objective_values), np.array(best_theta_total)



    def first_stage1(self, X, K):
        thetas = []
        objective_values = []
        best_theta_total = []
        for x in X:
            # 随机生成 K 个解
            solutions = np.random.uniform(lowbound, upbound, size=(K, theta_dim)) 
            objectives = min([objective_function(theta, x) for theta in solutions])
            thetas.append(solutions)
            best_theta = solutions[np.argmin(objectives)]  
            best_theta_total.append(best_theta)
            objective_values.append(objectives)
        return  np.array(thetas), np.array(objective_values), np.array(best_theta_total)
    

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



        

class OnlineApplication:
    def __init__(self, surface_model,quantile_models,quantile_points):
        self.surface_model = surface_model
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

        # Predict all quantile values for the first target variable
        quantile_predictions = np.array([
            model.predict(test_x[:, :fit_models[0].n_features_in_]) for model in fit_models
        ])  # Shape: (ntau, n_test)

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

        # Predict all quantile values in a single step for efficiency
        quantile_predictions = np.array([
            model.predict(test_x[:, :fit_models[0].n_features_in_]) for model in fit_models
        ])  # Shape: (ntau, n_test)

        for i in range(n_test):
            # Extract predictions for the current sample
            low_pred = quantile_predictions[low_indices[i], i]
            up_pred = quantile_predictions[up_indices[i], i]
            # Linearly interpolate to generate the sample
            gen_y[i] = low_pred + weights[i] * (up_pred - low_pred)

        return gen_y


    def generate_solutions(self, x, M):
        # Step 1: Generate solutions

        input_dim = x.shape[0]
        output_dim = len(self.quantile_models)
        generated_data = np.zeros((M, output_dim))

        test_x = np.tile(x, (M, 1))  # Initial input is just x_star
        generated_data[:, 0] = self.predicting_x0(test_x, self.quantile_models[0], self.quantile_points)

        # Predict subsequent target variables
        for d in range(1, output_dim):
            test_x = np.hstack([np.tile(x, (M, 1)), generated_data[:, :d]])
            generated_data[:, d] = self.predicting(test_x, self.quantile_models[d], self.quantile_points)
        generated_data = project(generated_data, lowbound=lowbound, upbound=upbound)
        return generated_data

    def generate_solutions1(self, x, M):

        d = len(self.quantile_models)
        Y_samples = np.zeros((M, d))

        for k in range(M):
            u = np.random.uniform(0, 1, d)
            for l in range(d):
                quantiles = self.quantile_points
                predictions = [model.predict(x)[0] for model in models[l]]
                interpolator = interp1d(quantiles, predictions, kind='linear', fill_value="extrapolate")
                Y_samples[k, l] = interpolator(u[l])
        generated_data = project(Y_samples, lowbound=lowbound, upbound=upbound)
        return generated_data

    def select_best_solution(self, x, solutions):
        x_flattened = x.flatten()
        # 将 solutions 与 x_flattened 合并成矩阵
        nn_inputs = np.hstack([solutions, np.tile(x_flattened, (solutions.shape[0], 1))])

        # 批量计算 K_star 矩阵
        K_star_batch = K_matrix(nn_inputs, self.surface_model.data_x, self.surface_model.kernel_function)

        # 批量计算 scores
        scores = K_star_batch @ self.surface_model.cov_matrix_inv @ self.surface_model.data_y


        # scores = []
        # for solution in solutions:
        #     nn_input = torch.tensor(np.concatenate([solution, x.flatten()]), dtype=torch.float32).unsqueeze(0)
        #     K_star = K_matrix(nn_input.numpy(), self.surface_model.data_x, self.surface_model.kernel_function)
        #     score = K_star @ self.surface_model.cov_matrix_inv @ self.surface_model.data_y
            
            
        #     scores.append(score)
        best_solution_idx = np.argmin(scores)
        return solutions[best_solution_idx]


    def optimize(self, x, M):
        solutions = self.generate_solutions(x, M)
        best_solution = self.select_best_solution(x, solutions)
        return best_solution


def objective_function(theta, x):
    
    theta, x = theta.reshape(1,theta_dim),x.reshape(covariate_dim,1)
    #print(theta.shape, A.shape, x.shape)
    hat_x = A @ x
    f = np.sum(np.cos(2 * theta) + hat_x * np.sin(2 * theta) + 0.1 * theta**2)
    return f

def gradient(theta, x):
    hat_x = A @ x
    grad = -2 * np.sin(2 * theta) + hat_x * 2 * np.cos(2 * theta) + 0.2*theta
    
    return grad 
  
def plot_theta_distribution1(x, y,x_new):
  
    K, d = y.shape  # K是解的数量，d是theta的维度

    # 创建图形
    sns.set_theme()
    if d == 1:
        fig, axes = plt.subplots(1, d, figsize=(12, 8))
    else:
        fig, axes = plt.subplots(2, int(d / 2), figsize=(12, 8))
        axes = axes.flatten() 
    if d == 1:
        axes = [axes]  # 如果只有一维，确保axes是一个列表
    
    # 绘制每个维度的分布
    for i in range(d):
        # 绘制真实的 θ 的分布
        sns.histplot(x[:, i], bins=100, kde=True, ax=axes[i], color='blue', stat='density', alpha=0.7, label='True-distribution', fill=True)
        sns.histplot(y[:, i], bins=100, kde=True, ax=axes[i], color='red', stat='density', alpha=0.7, label='Generate-distribution', fill=True)
        #sns.histplot(z[:, i], bins=100, kde=True, ax=axes[i], color='green', stat='density',alpha=0.7, label='Generate-CWGAN', fill=True)
        axes[i].set_title(f'Dimension {i+1}')
        axes[i].set_xlabel(r'$\theta{}$'.format(i+1))
        axes[i].set_ylabel('Density')
        axes[i].legend()

    plt.tight_layout()
    plt.savefig('distribution.pdf', format='pdf')
    plt.show()


    if d == 2:
        theta1_values = np.linspace(lowbound, upbound, 100)  # 生成 theta1 的取值范围
        theta2_values = np.linspace(lowbound, upbound, 100)  # 生成 theta2 的取值范围
        theta1_grid, theta2_grid = np.meshgrid(theta1_values, theta2_values)
        x0 = x_new
        objective_values = np.zeros_like(theta1_grid)
        for i in range(len(theta1_values)):
            for j in range(len(theta2_values)):
                theta = np.array([theta1_grid[i, j], theta2_grid[i, j]])
                objective_values[i, j] = objective_function(theta, x0)


    
        plt.figure(figsize=(8, 6))
        contour = plt.contour(theta1_grid, theta2_grid, objective_values, 20, cmap='viridis')
        plt.colorbar(contour)
        # 绘制真实的 θ 的散点图
        sns.scatterplot(x=x[:, 0], y=x[:, 1], color='blue', s=5, alpha=0.7, label='True')
        # 绘制生成的 θ 的散点图
        sns.scatterplot(x=y[:, 0], y=y[:, 1], color='red', s=5, alpha=0.7, label='Generate-GMM')
        #sns.scatterplot(x=z[:, 0], y=z[:, 1], color='green', s=5, alpha=0.7, label='Generate-CWGAN')
        plt.title('Scatter plot of two dimensions')
        plt.xlabel(r'$\theta_1$')
        plt.ylabel(r'$\theta_2$')
        plt.legend()
        plt.savefig('Scatter.pdf', format='pdf')
        plt.show()


class exp_kernel:
    def __init__(self, length=1, sigma_f=1):
        self.length = length
        self.sigma_f = sigma_f

    def __call__(self, x1, x2):
        y = np.linalg.norm(x1 - x2)
        return float(self.sigma_f * np.exp(-y**2 / (2 * self.length**2)))

class KRR_Offline:
    def __init__(self, data_x, data_y, kernel_function, lambda1=1e-7):
        self.data_x = data_x
        self.data_y = data_y
        self.kernel_function = kernel_function
        self.lambda1 = lambda1
        self.cov_matrix_inv = None

    def _compute_cov_matrix_inv(self, lambda1=None):
        if lambda1 is None:
            lambda1 = self.lambda1
        K = K_matrix(self.data_x, self.data_x, self.kernel_function)
        regularization = lambda1 * len(self.data_x) * np.identity(len(self.data_x))
        return np.linalg.inv(K + regularization)

    def cross_validate(self, lambda_list, num_folds=5):
        best_lambda = None
        best_error = float('inf')

        for lambda1 in lambda_list:
            kf = KFold(n_splits=num_folds)
            errors = []

            for train_index, val_index in kf.split(self.data_x):
                train_x, val_x = self.data_x[train_index], self.data_x[val_index]
                train_y, val_y = self.data_y[train_index], self.data_y[val_index]

                # Train the model on the training split
                K_train = K_matrix(train_x, train_x, self.kernel_function)
                regularization = lambda1 * len(train_x) * np.identity(len(train_x))
                
                cov_matrix_inv = np.linalg.inv(K_train + regularization)

                # Predict on the validation split
                K_val = K_matrix(val_x, train_x, self.kernel_function)
                predictions = K_val @ cov_matrix_inv @ train_y

                # Calculate mean squared error
                mse = np.mean((predictions - val_y) ** 2)
                errors.append(mse)

            avg_error = np.mean(errors)
            print(f"Lambda: {lambda1}, Average cross-validation error: {avg_error}")

            if avg_error < best_error:
                best_error = avg_error
                best_lambda = lambda1

        self.lambda1 = best_lambda
        self.cov_matrix_inv = self._compute_cov_matrix_inv(best_lambda)
        print(f"Best lambda: {best_lambda} with error: {best_error}")
        return best_lambda, best_error


class KRR_Online:
    def __init__(self, offline_model, online_x):
        self.offline_model = offline_model
        self.online_x = online_x

    def predict(self):
        K_star = K_matrix(self.online_x, self.offline_model.data_x, self.offline_model.kernel_function)
        predictions = K_star @ self.offline_model.cov_matrix_inv @ self.offline_model.data_y
        return predictions

# Supporting functions
def K_matrix(x1, x2, kernel_function):
    n1, n2 = len(x1), len(x2)
    K = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            K[i, j] = kernel_function(x1[i], x2[j])
    return K

def solver(x):
    num_initial_points = 10* theta_dim  
    initial_points = np.random.uniform(lowbound, upbound, size=(num_initial_points,theta_dim ))  # 随机采样初始点

    best_solution = None
    best_value = float('inf')

    
    for theta0 in initial_points:
        result = minimize(objective_function, theta0, args=(x,), method='L-BFGS-B') 

        if result.fun < best_value:
            best_value = result.fun
            best_solution = result.x

    print("Global minimum:", best_solution)
    print("Minimum value:", best_value)
    return best_solution,best_value


if __name__ == '__main__':

    np.random.seed(42)
    lowbound = 0
    upbound = 10
    eta = 0.01
    n = 500
    m = 100 
    K = 100
    covariate_dim = 2  
    theta_dim = 2  
    M = 5000
    #A = np.full((theta_dim, covariate_dim), 1)
    A = np.eye(theta_dim, covariate_dim)
    covariates = grid_sample(covariate_dim, lowbound, upbound, n)
    print(covariates.shape)

    offline_model = OfflineModel(n, m, K, covariate_dim, theta_dim)
    objective_values, thetas = offline_model.first_stage2(covariates, K)
    
    combined = np.hstack((thetas ,covariates))
    
    kernel_function = exp_kernel(length=1.0, sigma_f=1.0)
    lambda_list = [1e-1, 1e-2, 1e-3, 1e-4]

    offline_model_krr = KRR_Offline(combined, objective_values, kernel_function)
    best_lambda, best_error = offline_model_krr.cross_validate(lambda_list, num_folds=5)
    print(best_lambda, best_error)

    #nn_model = train_nn_surface_model(covariates, thetas, objective_values, covariate_dim)


    models, quantile_grid = offline_model.train_multi_output_qrgmm(covariates, thetas, m)

    online_app = OnlineApplication(offline_model_krr,models, quantile_grid)
    
    num_test_points = 20
    test_points = np.random.uniform(low=lowbound, high=upbound, size=(num_test_points, covariate_dim))
    fake_QRGMM ,true,true1 = [], [], []
    for x_new in test_points:
        
        best_solution = online_app.optimize(x_new, M)
        fake_QRGMM.append(objective_function(best_solution,x_new))
        
        best_solution, globalmin_objection = solver(x_new)
        true.append(globalmin_objection)

    solutions = online_app.generate_solutions(x_new, M)
    plot_theta_distribution1(thetas,solutions,x_new)

    df = pd.DataFrame({
    
    'Solver': true,
    'QRGMM': fake_QRGMM})

    sns.set_theme()
    df.plot(kind='line', marker='o', linewidth=2)
    
    plt.title('Covariate Dimension:{},Solution Dimension:{}'.format(covariate_dim,theta_dim))
    plt.ylabel('Objective Function')
    plt.xlabel('Test Points number')
    plt.legend(title="Models")
    plt.grid(True)
    plt.savefig('qrgmm.pdf', format='pdf')
    plt.show()
   
    
    





    #example
    solutions_distribution = np.zeros((M,theta_dim))
    x_new = x_new.reshape(1,covariate_dim)
    for i in range(M):
        solutions_distribution[i] = offline_model.first_stage2(x_new, K)[1]
    plot_theta_distribution1(solutions_distribution,solutions,x_new)





