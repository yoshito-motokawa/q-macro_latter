import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

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
param = {
    'sigma': 1.5,
    'beta': 0.98,
    'rho': 0.6,
    'sigma_eps': 0.6,
    'a_l': 0,
    'a_u': 20,
    'NA': 401,
    'NH': 2,
    'mu_h': -0.7,
}

# 生産性グリッドと遷移確率の計算
param['pi'], param['h'] = tauchen(param['NH'], param['mu_h'], param['rho'], param['sigma_eps'])

# 価格設定
r, w = 0.04, 1

# 家計問題を解く
aplus, c = solve_household(param, r, w)

# グラフ描画
a = np.linspace(param['a_l'], param['a_u'], param['NA'])
plt.figure(figsize=(10, 6))
plt.plot(a, aplus[:, 0] / (c[:, 0] + aplus[:, 0]), label='Low productivity (h_L)')
plt.plot(a, aplus[:, 1] / (c[:, 1] + aplus[:, 1]), label='High productivity (h_H)')
plt.xlabel('Current Assets')
plt.ylabel('Savings Rate')
plt.title('Savings Rate vs Current Assets')
plt.legend()
plt.grid(True)
plt.show()