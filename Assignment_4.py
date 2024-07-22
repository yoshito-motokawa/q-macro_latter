import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# tauchen関数
def tauchen(n, mu, rho, sigma):
    m = 1 / np.sqrt(1 - rho**2)
    state_space = np.linspace(mu - m*sigma, mu + m*sigma, n)
    d = (state_space[n-1] - state_space[0]) / (n-1)
    transition_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if j == 0:
                transition_matrix[i, 0] = norm.cdf((state_space[0] - rho*state_space[i] + d/2)/sigma)
            elif j == n-1:
                transition_matrix[i, n-1] = 1.0 - norm.cdf((state_space[n-1] - rho*state_space[i] - d/2)/sigma)
            else:
                transition_matrix[i, j] = norm.cdf((state_space[j] - rho*state_space[i] + d/2)/sigma) - norm.cdf((state_space[j] - rho*state_space[i] - d/2)/sigma)
    
    return transition_matrix, np.exp(state_space)

def solve_household(param, r, w):
    NA, NH = param['NA'], param['NH']
    h, a_l, a_u = param['h'], param['a_l'], param['a_u']
    sigma, beta, pi = param['sigma'], param['beta'], param['pi']
    
    a = np.linspace(a_l, a_u, NA)
    util = np.full((NA, NA, NH), -np.inf)
    
    for ia in range(NA):
        for ih in range(NH):
            for iap in range(NA):
                cons = w*h[ih] + (1.0 + r)*a[ia] - a[iap]
                if cons > 0:
                    util[iap, ia, ih] = cons**(1.0-sigma)/(1.0-sigma)
    
    v = np.zeros((NA, NH))
    v_new = np.zeros((NA, NH))
    iaplus = np.full((NA, NH), -1)
    
    tol = 1e-6
    while True:
        for ia in range(NA):
            for ih in range(NH):
                reward = util[:, ia, ih] + beta * (pi[ih, 0] * v[:, 0] + pi[ih, 1] * v[:, 1])
                v_new[ia, ih] = np.max(reward)
                iaplus[ia, ih] = np.argmax(reward)
        
        if np.max(np.abs(v_new - v)) < tol:
            break
        v = v_new.copy()
    
    aplus = a[iaplus]
    c = w * h[np.newaxis, :] + (1.0 + r) * a[:, np.newaxis] - aplus
    
    return aplus, c

# パラメータ設定
param_base = {
    'sigma': 1.5,
    'beta': 0.98,  # 元の時間選好率
    'rho': 0.6,
    'sigma_eps': 0.6,
    'a_l': 0,
    'a_u': 20,
    'NA': 401,
    'NH': 2,
    'mu_h': -0.7,
}

param_low_beta = param_base.copy()
param_low_beta['beta'] = 0.1  # 低下後の時間選好率

# 生産性グリッドと遷移確率の計算（両方のケースで同じ）
param_base['pi'], param_base['h'] = tauchen(param_base['NH'], param_base['mu_h'], param_base['rho'], param_base['sigma_eps'])
param_low_beta['pi'], param_low_beta['h'] = param_base['pi'], param_base['h']

# 価格設定
r, w = 0.04, 1

# β=0.98の場合の家計問題を解く
aplus_base, c_base = solve_household(param_base, r, w)

# β=0.1の場合の家計問題を解く
aplus_low_beta, c_low_beta = solve_household(param_low_beta, r, w)

# グラフ描画
a = np.linspace(param_base['a_l'], param_base['a_u'], param_base['NA'])
plt.figure(figsize=(12, 6))

# β=0.98の場合
plt.plot(a, aplus_base[:, 0] / (c_base[:, 0] + aplus_base[:, 0]), label='Low productivity (β=0.98)', color='blue')
plt.plot(a, aplus_base[:, 1] / (c_base[:, 1] + aplus_base[:, 1]), label='High productivity (β=0.98)', color='red')

# β=0.1の場合
plt.plot(a, aplus_low_beta[:, 0] / (c_low_beta[:, 0] + aplus_low_beta[:, 0]), label='Low productivity (β=0.1)', color='blue', linestyle='--')
plt.plot(a, aplus_low_beta[:, 1] / (c_low_beta[:, 1] + aplus_low_beta[:, 1]), label='High productivity (β=0.1)', color='red', linestyle='--')

plt.xlabel('Current Assets')
plt.ylabel('Savings Rate')
plt.title('Savings Rate vs Current Assets (Before and After β decrease)')
plt.legend()
plt.grid(True)
plt.show()