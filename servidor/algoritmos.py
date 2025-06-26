import numpy as np

def calcular_ganho_sinal(g):
    S, N = g.shape
    for l in range(S):
        ganho = 100 + (1/20) * np.sqrt(l * l)
        g[l, :] *= ganho
    return g

def fator_reducao(H):
    return np.linalg.norm(H.T @ H, 2)

def regularizacao(H, g):
    return np.max(np.abs(H.T @ g)) * 0.10

def erro(r_atual, r_anterior):
    return np.linalg.norm(r_atual)**2 - np.linalg.norm(r_anterior)**2

def cgne(H, g, epsilon=1e-10):
    f = np.zeros((H.shape[1], 1))
    r = g - H @ f
    p = H.T @ r
    i = 0
    while True:
        alpha = (r.T @ r) / (p.T @ p)
        f = f + alpha * p
        r_new = r - alpha * (H @ p)
        if np.linalg.norm(r_new) < epsilon:
            break
        beta = (r_new.T @ r_new) / (r.T @ r)
        p = H.T @ r_new + beta * p
        r = r_new
        i += 1
    return f, i

def cgnr(H, g, epsilon=1e-10):
    f = np.zeros((H.shape[1], 1))
    r = g - H @ f
    z = H.T @ r
    p = z
    i = 0
    while True:
        w = H @ p
        alpha = (z.T @ z) / (w.T @ w)
        f = f + alpha * p
        r = r - alpha * w
        z_new = H.T @ r
        if np.linalg.norm(r) < epsilon:
            break
        beta = (z_new.T @ z_new) / (z.T @ z)
        p = z_new + beta * p
        z = z_new
        i += 1
    return f, i
