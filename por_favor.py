import numpy as np
import os
import matplotlib.pyplot as plt

def cgne(H, g, max_iter=100, tol=1e-4):
    f = np.zeros(H.shape[1])
    r = g - H @ f
    p = H.T @ r
    for i in range(max_iter):
        alpha = np.dot(r, r) / np.dot(p, p)
        f = f + alpha * p
        r_next = r - alpha * (H @ p)
        if np.abs(np.linalg.norm(r) - np.linalg.norm(r_next)) < tol:
            break
        beta = np.dot(r_next, r_next) / np.dot(r, r)
        p = beta * p + H.T @ r_next
        r = r_next
    return f

# Caminhos relativos à pasta data
data_dir = os.path.dirname(__file__)
H_path = os.path.join(data_dir, 'H_1.npy')
g_path = os.path.join(data_dir, 'G_1.npy')

if not os.path.exists(H_path):
    raise FileNotFoundError(f"Arquivo de matriz H não encontrado: {H_path}")
if not os.path.exists(g_path):
    raise FileNotFoundError(f"Arquivo de vetor g não encontrado: {g_path}")

H = np.load(H_path)
g = np.load(g_path)

if len(H.shape) != 2:
    raise ValueError(f"A matriz H deve ser 2D, shape atual: {H.shape}")
if g.shape[0] != H.shape[0]:
    raise ValueError(f"O vetor g deve ter o mesmo número de linhas que H. g: {g.shape}, H: {H.shape}")

f = cgne(H, g)

# Tenta converter f em imagem quadrada
lado = int(np.sqrt(f.size))
if lado * lado != f.size:
    raise ValueError(f"O vetor f não pode ser convertido para imagem quadrada. Tamanho: {f.size}")
imagem = f.reshape((lado, lado))

plt.imshow(imagem, cmap='gray')
plt.title('Imagem reconstruída (CGNE)')
plt.show()