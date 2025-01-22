

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
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Offline Stage
import numpy as np
from sklearn.linear_model import QuantileRegressor
from scipy.interpolate import interp1d
plt.rcParams['figure.max_open_warning'] = 150  # 设置警告阈值为 50


def objective_function(theta, x):
    hat_x = A @ x
    f = np.sum(np.cos(2 * theta) + hat_x * np.sin(2 * theta) + 0.5 * theta**2)
    return f

def gradient(theta, x):
    hat_x = A @ x
    grad = -2 * np.sin(2 * theta) + hat_x * 2 * np.cos(2 * theta) + theta
    noise = np.random.normal(0, noise_std, d_theta)
    return grad + noise



def grid_sample(d, lowbound, upbound, point):
    n = int(round(point ** (1.0 / d))) +1 
    grid_1d = np.linspace(lowbound + 0.1, upbound - 0.1, n)
    grid_points = np.array(list(product(grid_1d, repeat=d)))
    if len(grid_points) > point:
        indices = np.random.choice(len(grid_points), size=point, replace=False)
        grid_points = grid_points[indices]
    return grid_points

def project(theta, lowbound, upbound):
 
    return np.clip(theta, lowbound, upbound)


def first_stage(X, x_dim, K):
    thetas = []
    objective_values = []
    best_theta_total = []
    for x in X:
        #solutions = random_points(K, x_dim) # 随机生成 K 个解
        solutions = np.random.uniform(lowbound, upbound, size=(K, x_dim)) 
        objectives = min([objective_function(theta, x) for theta in solutions])
        thetas.append(solutions)
        best_theta = solutions[np.argmin(objectives)]  # 选择最优解
        best_theta_total.append(best_theta)
        objective_values.append(objectives)
    return X, np.array(thetas), np.array(objective_values), np.array(best_theta_total)













if __name__ == '__main__':
    np.random.seed(42)
    lowbound = -4
    upbound = 4
    eta = 0.1  
    covariate_dim = 5
    n = 100
