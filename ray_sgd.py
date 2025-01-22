

from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import gaussian_kde
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
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Offline Stage
import numpy as np
from sklearn.linear_model import QuantileRegressor
from scipy.interpolate import interp1d
plt.rcParams['figure.max_open_warning'] = 150  # 设置警告阈值为 50
from torch.utils.data import DataLoader, TensorDataset
import ray
ray.init(num_cpus=20)  

def grid_sample( d, lowbound, upbound, point):
        n = int(round(point ** (1.0 / d))) +1 
        grid_1d = np.linspace(lowbound + 0.1, upbound - 0.1, n)
        grid_points = np.array(list(product(grid_1d, repeat=d)))
        if len(grid_points) > point:
            indices = np.random.choice(len(grid_points), size=point, replace=False)
            grid_points = grid_points[indices]
        return grid_points


class SurfaceModel(nn.Module):
    def __init__(self, input_dim):
        super(SurfaceModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 输出单个目标值
        )
    
    def forward(self, x):
        return self.net(x)

# def train_nn_surface_model(X, thetas, objective_values, x_dim, epochs=150, lr=1e-3, batch_size=16):
    
#     n, K, theta_dim = thetas.shape
    
#     X_repeated = np.repeat(X, K, axis=0)  # 将 X 重复 K 次
#     thetas_flat = thetas.reshape(-1, theta_dim)  # 展平 thetas
#     inputs = np.hstack([thetas_flat, X_repeated])  # 拼接 theta 和 x
#     labels = objective_values.flatten()


#     inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
#     labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

#     # 定义神经网络模型
#     model = SurfaceModel(input_dim=theta_dim + x_dim)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.MSELoss()

#     # 训练模型
#     for epoch in range(epochs):
#         model.train()
#         permutation = torch.randperm(inputs_tensor.size(0))
#         for i in range(0, inputs_tensor.size(0), batch_size):
#             indices = permutation[i:i+batch_size]
#             batch_inputs = inputs_tensor[indices]
#             batch_labels = labels_tensor[indices]

#             # 前向传播
#             predictions = model(batch_inputs)
#             loss = criterion(predictions, batch_labels)

#             # 反向传播
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         if (epoch + 1) % 10 == 0:
#             print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

#     return model

def train_nn_surface_model(X, thetas, objective_values, x_dim, epochs=150, lr=1e-3, batch_size=16):
    # 检查是否有可用的 GPU
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    n, K, theta_dim = thetas.shape

    # 数据预处理
    X_repeated = np.repeat(X, K, axis=0)  # 将 X 重复 K 次
    thetas_flat = thetas.reshape(-1, theta_dim)  # 展平 thetas
    inputs = np.hstack([thetas_flat, X_repeated])  # 拼接 theta 和 x
    labels = objective_values.flatten()

    # 将数据转换为 Tensor 并移动到设备
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1).to(device)

    # 定义神经网络模型并移动到设备
    model = SurfaceModel(input_dim=theta_dim + x_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 训练模型
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(inputs_tensor.size(0)).to(device)  # 确保在 GPU 上生成
        for i in range(0, inputs_tensor.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_inputs = inputs_tensor[indices]
            batch_labels = labels_tensor[indices]

            # 前向传播
            predictions = model(batch_inputs)
            loss = criterion(predictions, batch_labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    return model

def project(theta, lowbound, upbound):

    return np.clip(theta, lowbound, upbound)

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

@ray.remote
def train_model_for_tau(X, Y, tau, hidden_dim=64, lr=0.001, epochs=100, batch_size=32):
   
    
    # 转换数据为 PyTorch 张量
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
    def __init__(self, n, m, K, covariate_dim, theta_dim):
        self.n = n  # Number of covariate points
        self.m = m  # Number of quantiles
        self.K = K  # Number of samples per covariate point
        self.covariate_dim = covariate_dim  # Dimension of covariates
        self.theta_dim = theta_dim  # Dimension of theta
        self.surface_model = None
        self.quantile_models = []

    #random search
    def first_stage(self, X, K):
        thetas = []
        objective_values = []
        best_theta_total = []
        for x in X:
            solutions = np.random.uniform(lowbound, upbound, size=(K, theta_dim)) 
            objectives = [objective_function(theta, x) for theta in solutions]
            #objectives = min(objectives1)
            thetas.append(solutions)
            best_theta = solutions[np.argmin(objectives)]  
            best_theta_total.append(best_theta)
            objective_values.append(objectives)
            #print(objective_function(best_theta, x),objectives)
        return   np.array(thetas),np.array(objective_values), np.array(best_theta_total)


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
             
            solutions =  np.array(theta_av)
            theta_av = np.mean(solutions, axis=0)   
            #best_theta_total.append(theta)
            
            objectives = [objective_function(theta, x) for theta in solutions]
            thetas1.append(solutions)

            #best_theta = np.mean(np.array(theta_av),axis = 0)

            best_theta_total.append(theta_av)
            objective_values.append(objectives)
        return  np.array(thetas1),np.array(objective_values), np.array(best_theta_total)



    # def train_multi_output_qrgmm(self,X,Y,m):
        
    #     n, d = Y.shape
    #     quantile_grid = np.linspace(1/m, (m-1)/m, m-1)
    #     models = []

    #     for l in range(d):
    #         models_l = []
    #         for tau in quantile_grid:
    #             model = QuantileRegressor(quantile=tau, solver="highs")
    #             model.fit(X, Y[:, l])
    #             models_l.append(model)
    #         models.append(models_l)
    #     return models, quantile_grid


    # def train_multi_output_qrgmm(self,X, Y, m, hidden_dim=64, lr=0.001, epochs=100, batch_size=32):
            
    #     n, d = Y.shape
    #     quantile_grid = torch.linspace(1/m, (m-1)/m, m-1)
    #     models = []

    #     # Convert data to PyTorch tensors
    #     X_tensor = torch.tensor(X, dtype=torch.float32)
    #     Y_tensor = torch.tensor(Y, dtype=torch.float32)

    #     for l in range(d):
    #         models_l = []
    #         for tau in quantile_grid:
    #             model = QuantileRegressionNN(input_dim=X.shape[1], hidden_dim=hidden_dim)
    #             optimizer = optim.Adam(model.parameters(), lr=lr)
                
    #             def quantile_loss(y_pred, y_true, tau):
    #                 errors = y_true - y_pred
    #                 return torch.mean(torch.max(tau * errors, (tau - 1) * errors))

    #             # Training loop
    #             for epoch in range(epochs):
    #                 model.train()
    #                 permutation = torch.randperm(X_tensor.size(0))

    #                 for i in range(0, X_tensor.size(0), batch_size):
    #                     indices = permutation[i:i + batch_size]
    #                     batch_X, batch_Y = X_tensor[indices], Y_tensor[indices, l:l+1]

    #                     optimizer.zero_grad()
    #                     outputs = model(batch_X)
    #                     loss = quantile_loss(outputs, batch_Y, tau)
    #                     loss.backward()
    #                     optimizer.step()

    #             models_l.append(model)
    #         models.append(models_l)

    #     return models, quantile_grid.numpy()

    def train_multi_output_qrgmm(self, X, Y, m, hidden_dim=64, lr=0.001, epochs=150, batch_size=32):
    
        n, d = Y.shape
        quantile_grid = torch.linspace(1/m, (m-1)/m, m-1).numpy()

        

        all_models = []
        for l in range(d):
            # 为每个目标变量并行训练多个分位数模型
            results = ray.get([
                train_model_for_tau.remote(
                    X, Y[:, l:l+1], tau, hidden_dim, lr, epochs, batch_size
                )
                for tau in quantile_grid
            ])
            all_models.append(results)

        
        

        return all_models, quantile_grid


        

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
        # Evaluate solutions
       
        scores = []
        for solution in solutions:
            nn_input = torch.tensor(np.concatenate([solution, x.flatten()]), dtype=torch.float32).unsqueeze(0)
            score = self.surface_model(nn_input).item()
            scores.append(score)
        
 
        best_solution_idx = np.argmin(scores)
        return solutions[best_solution_idx]





    def optimize(self, x, M):
        #solutions = self.generate_solutions(x, M)
        solutions = self.generate_solutions(x, M)
       
        best_solution = self.select_best_solution(x, solutions)
        return best_solution


def objective_function(theta, x):
    
    theta, x = theta.reshape(1,theta_dim),x.reshape(covariate_dim,1)
    
    hat_x = A @ x
    f = np.sum(np.cos(2 * theta) + hat_x * np.sin(2 * theta) + 0.1 * theta**2)
    return f

def gradient(theta, x):
    hat_x = A @ x
    grad = -2 * np.sin(2 * theta) + hat_x * 2 * np.cos(2 * theta) + 0.2*theta
    noise = np.random.normal(0, std, theta_dim)
    grad = grad + noise
    return grad 
  
def plot_theta_distribution1(x, y,x_new):
    K, d = y.shape  
    sns.set_theme()
    if d == 1:
        fig, axes = plt.subplots(1, d, figsize=(12, 8))
    else:
        fig, axes = plt.subplots(2, int(d / 2), figsize=(12, 8))
        axes = axes.flatten() 
    if d == 1:
        axes = [axes]  
    
    # 绘制每个维度的分布
    for i in range(d):
        kde = gaussian_kde(x[:, i])
        x_vals = np.linspace(x[:, i].min(), x[:, i].max(), 500)  # KDE 的 x 轴范围
        y_vals = kde(x_vals)  # KDE 计算结果
        axes[i].hist(x[:, i], bins=100, color='blue', alpha=0.7, density=True,label='True-distribution')
        axes[i].plot(x_vals, y_vals, color='blue', alpha=0.7)
       
        kde = gaussian_kde(y[:, i])
        x_vals = np.linspace(y[:, i].min(), y[:, i].max(), 500)  # KDE 的 x 轴范围
        y_vals = kde(x_vals)  # KDE 计算结果
        axes[i].hist(y[:, i], bins=100, color='red', alpha=0.7, density=True,label='QRGMM-distribution')
        axes[i].plot(x_vals, y_vals, color='red', alpha=0.7)
        # sns.histplot(x[:, i], bins=100, kde=True, ax=axes[i], color='blue', stat='density', alpha=0.7, label='True-distribution', fill=True)
        # sns.histplot(y[:, i], bins=100, kde=True, ax=axes[i], color='red', stat='density', alpha=0.7, label='Generate-distribution', fill=True)
        axes[i].set_title(f'Dimension {i+1}')
        axes[i].set_xlabel(r'$\theta_{}$'.format(i+1))
        axes[i].set_ylabel('Density')
        axes[i].legend(title="Online Stage")
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
        sns.scatterplot(x=x[:, 0], y=x[:, 1], color='blue', s=5, alpha=0.7, label='True')
        sns.scatterplot(x=y[:, 0], y=y[:, 1], color='red', s=5, alpha=0.7, label='Generate-(QRGMM)')
        plt.title('contour plots')
        plt.xlabel(r'$\theta_1$')
        plt.ylabel(r'$\theta_2$')
        plt.legend(title="Online Stage")
        plt.savefig('Scatter.pdf', format='pdf')
        plt.tight_layout()
        plt.show()


class KRR_Offline:
    def __init__(self, data_x, data_y, kernel_function, lambda1=1e-7):
        self.data_x = data_x
        self.data_y = data_y
        self.kernel_function = kernel_function
        self.lambda1 = lambda1
        self.cov_matrix_inv = self._compute_cov_matrix_inv()

    def _compute_cov_matrix_inv(self):
        K = K_matrix(self.data_x, self.data_x, self.kernel_function)
        regularization = self.lambda1 * len(self.data_x) * np.identity(len(self.data_x))
        return np.linalg.inv(K + regularization)


class KRR_Online:
    def __init__(self, offline_model, online_x):
        self.offline_model = offline_model
        self.online_x = online_x

    def predict(self):
        K_star = K_matrix(self.online_x, self.offline_model.data_x, self.offline_model.kernel_function)
        predictions = K_star @ self.offline_model.cov_matrix_inv @ self.offline_model.data_y
        return predictions

# Supporting functions (to ensure the above classes work)
def K_matrix(x1, x2, kernel_function):
    n1, n2 = len(x1), len(x2)
    K = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            K[i, j] = kernel_function(x1[i], x2[j])
    return K

def exp_kernel():
    return lambda x1, x2: np.exp(-np.linalg.norm(x1 - x2)**2)

def pic_dis(data):
    plt.figure(figsize=(8, 6))
    sns.histplot(data, bins=20, kde=True, color='blue', alpha=0.7)
    plt.title("Distribution of Objective Function Values", fontsize=14)
    plt.xlabel("Objective Function Value", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(alpha=0.3)
    plt.show()

def solver(x):
    num_initial_points = 100* theta_dim  
    initial_points = np.random.uniform(lowbound, upbound, size=(num_initial_points,theta_dim ))  # 随机采样初始点

    best_solution = None
    best_value = float('inf')

    
    for theta0 in initial_points:
        result = minimize(objective_function, theta0, args=(x,), method='L-BFGS-B') 

        if result.fun < best_value:
            best_value = result.fun
            best_solution = result.x

    # print("Global minimum:", best_solution)
    # print("Minimum value:", best_value)
    return best_solution,best_value

def plot_x_y(covariates, thetas, xlabel='Covariates', ylabel='Thetas', title='Covariates vs Thetas'):
    
    plt.figure(figsize=(8, 6))
    plt.plot(covariates, thetas, marker='o', linestyle='-', color='b', label='y = f(x)')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()





if __name__ == '__main__':

    np.random.seed(42)
    std = 1
    lowbound = -5
    upbound = 5
    eta = 0.01 
    # n = 500
    m = 100 
    K = 25
    covariate_dim = 2
    theta_dim = 2  
    M = 5000 
    
    
    #A = np.eye(theta_dim, covariate_dim)
    # covariates = grid_sample(covariate_dim, lowbound, upbound, n)
    


    
    mse_method_d, mse_method2_d, mse_method3_d, mse_method4_d = [], [], [], []
    theta_dim1 = [2]
    for theta_dim in theta_dim1:
        n = 1000
        K = 20
        A = np.full((theta_dim, covariate_dim), 1)
        covariates = grid_sample(covariate_dim, lowbound, upbound, n)
        print(covariates.shape)
        offline_model = OfflineModel(n, m, K, covariate_dim, theta_dim)
        
        thetas,objective_values,best_theta = offline_model.first_stage2(covariates, K)
        
        #pic_dis(best_theta)
        models, quantile_grid = offline_model.train_multi_output_qrgmm(covariates, best_theta, m)

        print(covariates.shape, thetas.shape, objective_values.shape)
        
        
        
        
        start_time = time.time()
        nn_model = train_nn_surface_model(covariates, thetas, objective_values, covariate_dim)
        end_time = time.time()
        print(f"time: {end_time - start_time:.2f} s")
        
        online_app = OnlineApplication(nn_model,models, quantile_grid)
        
        
    #     num_test_points = 500
    #     test_points = np.random.uniform(low=lowbound, high=upbound, size=(num_test_points, covariate_dim))
    #     method1,method2,method3,method4 = (np.zeros(num_test_points) for _ in range(4))
        
    #     optimal = np.zeros(num_test_points)
        
    #     for i,x_new in enumerate(test_points):
    #         if i % 100 == 0:
    #             print(i)
            
    #         x_new = x_new.reshape(1, covariate_dim)
    #         the1 = np.random.uniform(low=lowbound, high=upbound, size=(1, theta_dim))
    #         method1[i] = objective_function(the1, x_new)
            
    #         the2 = offline_model.first_stage(x_new, K)[2]
            
    #         method2[i] = objective_function(the2, x_new)
    #         the3 = online_app.generate_solutions(x_new, 1)
    #         method3[i] = objective_function(the3, x_new)
    #         the4 = online_app.optimize(x_new, 10)
    #         method4[i] = objective_function(the4, x_new)
    #         optimal[i] = solver(x_new)[1]
        
    #     mse_method1 = np.mean(method1 - optimal) 
    #     mse_method2 = np.mean(method2 - optimal) 
    #     mse_method3 = np.mean(method3 - optimal) 
    #     mse_method4 = np.mean(method4 - optimal) 
    #     mse_method_d.append(mse_method1)
    #     mse_method2_d.append(mse_method2)
    #     mse_method3_d.append(mse_method3)
    #     mse_method4_d.append(mse_method4)
        
    #     print(mse_method1, mse_method2, mse_method3, mse_method4)
  
    # df = pd.DataFrame({
    
    # 'randomly generate': mse_method_d,
    # 'pure random search': mse_method2_d,
    # 'Generative 1 solution(QRGMM)': mse_method3_d,
    # 'Generative 10 solution(QRGMM)': mse_method4_d})
    
    
    # sns.set_theme()
    # df.plot(kind='line', marker='o', linewidth=2)
    
    # #plt.title('Covariate Dimension:{},Solution Dimension:{}'.format(covariate_dim,theta_dim))
    
    # plt.title(r'$d_x = {},d_\theta = {} $'.format(covariate_dim,theta_dim1), fontsize=15)
    
    
    # plt.ylabel(r'$f(\theta, x)$')
    # plt.xlabel('Test Points number')
    # plt.legend(title="Methods")
    # plt.grid(True)
    # plt.savefig('qrgmm.pdf', format='pdf')
    # plt.show()
     
     
     
     
     
     
     
     
     
     
    # num_test_points = 20
    # test_points = np.random.uniform(low=lowbound, high=upbound, size=(num_test_points, covariate_dim))
    # fake_QRGMM ,true,true1 = [], [], []
    # for x_new in test_points:
    #     best_solution = online_app.optimize(x_new, M)
    #     fake_QRGMM.append(objective_function(best_solution,x_new))
        
    #     best_solution, globalmin_objection = solver(x_new)
    #     true.append(globalmin_objection)
    

    # # # #example
    # solutions_distribution = np.zeros((M,theta_dim))
    # f = []
    # for i in range(M):
    #     solutions_distribution[i] = offline_model.first_stage(x_new, K)[1]
    #     f.append(objective_function(solutions_distribution[i],x_new))
    # best_f = min(f)
    # objective_function(best_solution,x_new)

    x_new = np.random.uniform(low=lowbound, high=upbound, size=(1, covariate_dim))
    solutions = online_app.generate_solutions(x_new, M)
    solutions_distribution = np.zeros((M,theta_dim))
    x_new = x_new.reshape(1,covariate_dim)
    for i in range(M):
        solutions_distribution[i] = offline_model.first_stage2(x_new, K)[2]
    plot_theta_distribution1(solutions_distribution,solutions,x_new)

    



    # df = pd.DataFrame({
    
    # 'Solver': true,
    # 'Our Generative Method': fake_QRGMM})

    # sns.set_theme()
    # df.plot(kind='line', marker='o', linewidth=2)
    
    # #plt.title('Covariate Dimension:{},Solution Dimension:{}'.format(covariate_dim,theta_dim))
    
    # plt.title(r'$d_x = {},d_\theta = {} $'.format(covariate_dim,theta_dim), fontsize=15)
    
    
    # plt.ylabel(r'$f(\theta, x)$')
    # plt.xlabel('Test Points number')
    # plt.legend(title="Methods")
    # plt.grid(True)
    # plt.savefig('qrgmm.pdf', format='pdf')
    # plt.show()
    
    ray.shutdown()