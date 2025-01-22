
from scipy.stats import gaussian_kde
from sklearn.metrics.pairwise import rbf_kernel
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
import time
from itertools import product
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
from sklearn.linear_model import QuantileRegressor
from scipy.interpolate import interp1d
plt.rcParams['figure.max_open_warning'] = 150  # 设置警告阈值为 50
from torch.utils.data import DataLoader, TensorDataset
import ray
ray.init(num_cpus=20)  



# class Generator(nn.Module):
#     def __init__(self, z_dim, x_dim, theta_dim):
#         super(Generator, self).__init__()
#         # 噪声和x的输入
#         self.fc1 = nn.Linear(z_dim + x_dim, 256)
#         self.fc2 = nn.Linear(256, 512)
#         self.fc3 = nn.Linear(512, theta_dim)

#     def forward(self, z, x):
#         # 将噪声z和条件x连接在一起
#         z_x = torch.cat([z, x], dim=-1)
#         out = torch.relu(self.fc1(z_x))
#         out = torch.relu(self.fc2(out))
#         theta = self.fc3(out)
#         return theta

# # Discriminator网络定义
# class Discriminator(nn.Module):
#     def __init__(self, theta_dim, x_dim):
#         super(Discriminator, self).__init__()
#         self.fc1 = nn.Linear(theta_dim + x_dim, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 1)

#     def forward(self, theta, x):
#         # 将x和theta连接在一起
#         x_theta = torch.cat([x, theta], dim=-1)
#         out = torch.relu(self.fc1(x_theta))
#         out = torch.relu(self.fc2(out))
#         validity = torch.sigmoid(self.fc3(out))  # 输出 [0, 1] 的值，表示真假
#         return validity
class Generator(nn.Module):
    def __init__(self, z_dim, x_dim, theta_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + x_dim, 128),
            nn.ReLU(),
            nn.Linear(128, theta_dim)
        )
    
    def forward(self, z, x):
        return self.net(torch.cat([z, x], dim=1))

class Discriminator(nn.Module):
    def __init__(self, theta_dim, x_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(theta_dim + x_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, theta, x):
        return self.net(torch.cat([theta, x], dim=1))


def gradient_penalty(D, real_data, fake_data, x):
    batch_size = real_data.size(0)
    
    # 确保 epsilon 在和 real_data 同一设备上
    epsilon = torch.rand(batch_size, 1, device=real_data.device)
    
    # 计算插值数据，并确保它支持梯度计算
    interpolated_data = epsilon * real_data + (1 - epsilon) * fake_data
    interpolated_data.requires_grad_(True)

    # 判别器输出
    x = x.to(real_data.device)  # 确保 x 在同一设备
    validity = D(interpolated_data, x)

    # 计算梯度
    gradients = torch.autograd.grad(
        outputs=validity,
        inputs=interpolated_data,
        grad_outputs=torch.ones_like(validity, device=real_data.device),
        create_graph=True,
        retain_graph=True
    )[0]

    # 计算梯度的 L2 范数
    gradients = gradients.view(batch_size, -1)  # 展平
    grad_l2_norm = torch.norm(gradients, p=2, dim=1)
    grad_penalty = torch.mean((grad_l2_norm - 1) ** 2)

    return grad_penalty


# CGAN训练函数
def train_cgan(X, thetas, z_dim, n_iterations, batch_size):
    x_dim = X.shape[1]
    theta_dim = thetas.shape[1]

    # 检查是否有 GPU 并设置设备
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 初始化 Generator 和 Discriminator 并移动到 GPU
    G = Generator(z_dim, x_dim, theta_dim).to(device)
    D = Discriminator(theta_dim, x_dim).to(device)

    # 定义优化器
    g_optimizer = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))

    # 转换为 Tensor 并移动到 GPU
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    thetas_tensor = torch.tensor(thetas, dtype=torch.float32).to(device)

    for iteration in range(n_iterations):
        for batch_idx in range(0, len(X), batch_size):
            # 获取当前批次数据
            x_batch = X_tensor[batch_idx:batch_idx + batch_size]
            theta_batch = thetas_tensor[batch_idx:batch_idx + batch_size]

            # 训练 Discriminator
            for _ in range(2):  # 判别器多次更新，以增强判别器的稳定性
                z = torch.randn(x_batch.size(0), z_dim, device=device)  # 在 GPU 上生成随机噪声
                fake_thetas = G(z, x_batch)

                D_real = D(theta_batch, x_batch)
                D_fake = D(fake_thetas.detach(), x_batch)

                # Wasserstein 损失
                d_loss = -torch.mean(D_real) + torch.mean(D_fake)

                # 添加梯度惩罚
                gp = gradient_penalty(D, theta_batch, fake_thetas, x_batch)
                d_loss += 20 * gp

                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()

            # 训练 Generator
            z = torch.randn(x_batch.size(0), z_dim, device=device)
            fake_thetas = G(z, x_batch)
            D_fake = D(fake_thetas, x_batch)

            # Wasserstein 损失
            g_loss = -torch.mean(D_fake)

            # Generator 的总损失
            lambda_theta, alpha = 0.8, 0.2
            loss_theta = torch.mean((fake_thetas - theta_batch) ** 2)
            total_loss_G = lambda_theta * loss_theta + alpha * g_loss

            g_optimizer.zero_grad()
            total_loss_G.backward()
            g_optimizer.step()

        # 打印损失
        if (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration + 1}, D Loss: {d_loss.item()}, G Loss: {total_loss_G.item()}")

    return G






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
        x = x.to(device)
        return self.net(x)

def train_nn_surface_model(X, thetas, objective_values, x_dim, epochs=300, lr=2*1e-3, batch_size=16):
    
    n,  theta_dim = thetas.shape
    X_repeated = X  # 将 X 重复 K 次
    thetas_flat = thetas.reshape(-1, theta_dim)  # 展平 thetas
    inputs = np.hstack([thetas_flat, X_repeated])  # 拼接 theta 和 x
    labels = objective_values.flatten()


    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    # 定义神经网络模型
    model = SurfaceModel(input_dim=theta_dim + x_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 训练模型
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(inputs_tensor.size(0))
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
    def random_search(self, X, K):
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
            objective_values.append(min(objectives))
        return   np.array(objective_values), np.array(best_theta_total)
    

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

        ray.shutdown()
        return all_models, quantile_grid


class OnlineApplication:
    def __init__(self, surface_model,quantile_models,quantile_points):
        self.surface_model = surface_model
        self.quantile_models = quantile_models
        self.quantile_points = quantile_points
        self.surface_model.to(device)  
        



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
    
    

        

    #slow
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
            nn_input = nn_input.to(device)
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

def finite_difference_gradient(f, theta, x, h=1e-5):
    theta = np.asarray(theta, dtype=float)
    n = len(theta)
    grad = np.zeros_like(theta)
    
    for i in range(n):
        theta_forward = theta.copy()
        theta_backward = theta.copy()
        theta_forward[i] += h
        theta_backward[i] -= h
        grad[i] = (f(theta_forward, x) - f(theta_backward, x)) / (2 * h)  # 中央差分
    return grad


def gradient(theta, x):
    
    return finite_difference_gradient(objective_function, theta, x)


# def gradient(theta, x):
#     hat_x = A @ x
#     grad = -2 * np.sin(2 * theta) + hat_x * 2 * np.cos(2 * theta) + 1*theta
#     return grad 
  
def plot_theta_distribution1(x, y,z,x_new):
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
        axes[i].hist(y[:, i], bins=100, color='red', alpha=0.7, density=True,label='Generate-distribution(QRGMM)')
        axes[i].plot(x_vals, y_vals, color='red', alpha=0.7)

        # kde = gaussian_kde(z[:, i])
        # x_vals = np.linspace(z[:, i].min(), z[:, i].max(), 500)  # KDE 的 x 轴范围
        # y_vals = kde(x_vals)  # KDE 计算结果
        # axes[i].hist(z[:, i], bins=20, color='green', alpha=0.7, density=True,label='Generate-distribution(CWGAN)')
        # axes[i].plot(x_vals, y_vals, color='green', alpha=0.7)

        # sns.histplot(x[:, i], bins=100, kde=True, ax=axes[i], color='blue', stat='density', alpha=0.7, label='True-distribution', fill=True)
        # sns.histplot(z[:, i], bins=100, kde=True, ax=axes[i], color='green', stat='density', alpha=0.7, label='CWGAN-distribution', fill=True)
        # sns.histplot(y[:, i], bins=100, kde=True, ax=axes[i], color='red', stat='density', alpha=0.7, label='QRGMM-distribution', fill=True)
        axes[i].set_title(f'Dimension {i+1}')
        axes[i].set_xlabel(r'$\theta_{}$'.format(i+1))
        axes[i].set_ylabel('Density')
        axes[i].legend(title="Online Stage")

    plt.tight_layout()
    
    plt.savefig('distribution_online_gan.pdf', format='pdf')
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
        sns.scatterplot(x=y[:, 0], y=y[:, 1], color='red', s=5, alpha=0.7, label='Generate-(QRGMM)')
        
        plt.title('contour plots')
        plt.xlabel(r'$\theta_1$')
        plt.ylabel(r'$\theta_2$')
        plt.legend(title="Online Stage")
        plt.savefig('Scatter_gan_online.pdf', format='pdf')
        plt.tight_layout()
        plt.show()

def plot_theta_distribution(x, y,z):
    #K, d = y.shape  
    print(x.shape,y.shape,z.shape)
    d = theta_dim
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
        axes[i].hist(y[:, i], bins=100, color='red', alpha=0.7, density=True,label='Generate-distribution(QRGMM)')
        axes[i].plot(x_vals, y_vals, color='red', alpha=0.7)

        kde = gaussian_kde(z[:, i])
        x_vals = np.linspace(z[:, i].min(), z[:, i].max(), 500)  # KDE 的 x 轴范围
        y_vals = kde(x_vals)  # KDE 计算结果
        axes[i].hist(z[:, i], bins=100, color='green', alpha=0.7, density=True,label='Generate-distribution(CWGAN)')
        axes[i].plot(x_vals, y_vals, color='green', alpha=0.7)


        # sns.histplot(x[:, i], bins=100, kde=True, ax=axes[i], color='blue', stat='density', alpha=0.7, label='True-distribution', fill=True)
        
        # sns.histplot(z[:, i], bins=100, kde=True, ax=axes[i], color='green', stat='density', alpha=0.7, label='CWGAN-distribution', fill=True)
        
        # sns.histplot(y[:, i], bins=100, kde=True, ax=axes[i], color='red', stat='density', alpha=0.7, label='QRGMM-distribution', fill=True)

        axes[i].set_title(f'Dimension {i+1}')
        axes[i].set_xlabel(r'$\theta_{}$'.format(i+1))
        axes[i].set_ylabel('Density')
        axes[i].legend(title="Offline Stage")

    plt.tight_layout()
    
    plt.savefig('distribution_off_gan.pdf', format='pdf')
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
    num_initial_points = 10* theta_dim  
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

if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    np.random.seed(42)
    lowbound = -5
    upbound = 5
    eta = 0.01 
    n = 500
    m = 100 
    K = 10
    covariate_dim = 2
    theta_dim = 2  
    M = 500
    A = np.full((theta_dim, covariate_dim), 1)
    
    #A = np.eye(theta_dim, covariate_dim)
    covariates = grid_sample(covariate_dim, lowbound, upbound, n)
    print(covariates.shape)


    offline_model = OfflineModel(n, m, K, covariate_dim, theta_dim)
    objective_values, thetas = offline_model.random_search(covariates, K)
    #pic_dis(thetas)
    models, quantile_grid = offline_model.train_multi_output_qrgmm(covariates, thetas, m)

    print(covariates.shape, thetas.shape, objective_values.shape)

    #####cgan
    z_dim = 10 
    n_iterations = 1000
    batch_size1 = 32
    G =  train_cgan(thetas, thetas, z_dim, n_iterations, batch_size1)
    print("Training completed. Generator is ready to generate samples conditioned on x.")
    candidates = []
    
    G = G.to(device)
    for x in covariates:
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  
        x_tensor = x_tensor.to(device)
        for _ in range(1):
            z = torch.randn(1, z_dim)  
            z = z.to(device)
            theta_g = G(z, x_tensor)  
            clip_theta = project(theta_g.detach().cpu().numpy().squeeze(), lowbound, upbound)
            candidates.append(clip_theta)
            
    solutions_cwgan = np.array(candidates)
    print(solutions_cwgan.shape)


    


    #nn_model = train_nn_surface_model(covariates, thetas, objective_values, covariate_dim)
    nn_model = None
    online_app = OnlineApplication(nn_model,models, quantile_grid)
    


######### online application
    x_new = np.random.uniform(low=lowbound, high=upbound, size=(1, covariate_dim))
    solutions = online_app.generate_solutions(x_new, M)
    solutions_distribution = np.zeros((M,theta_dim))
    x_new = x_new.reshape(1,covariate_dim)
    candidates = []
    for i in range(M):
        solutions_distribution[i] = offline_model.random_search(x_new, K)[1]
        z = torch.randn(1, z_dim)  
        z = z.to(device)
        x_tensor = torch.tensor(x_new, dtype=torch.float32) 
        x_tensor = x_tensor.to(device)
        theta_g = G(z, x_tensor)  
        clip_theta = project(theta_g.detach().cpu().numpy().squeeze(), lowbound, upbound)
        candidates.append(clip_theta)
    solutions_cwgan = np.array(candidates).reshape((M,theta_dim))
    print(solutions_cwgan.shape, solutions.shape,solutions_distribution.shape)
    plot_theta_distribution1(solutions_distribution,solutions,solutions_cwgan,x_new)
    