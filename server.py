# -----------------------------------------------------------------------------
# server.py - Servidor de Reconstrução de Imagem (v2.6)
# -----------------------------------------------------------------------------

import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, send_file, make_response
import logging
from numba import jit
import threading
import os
import io
from PIL import Image
import psutil
import time
import matplotlib.pyplot as plt

os.makedirs("log", exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(os.path.join("log", "server.log")), logging.StreamHandler()])

@jit(nopython=True)
def apply_signal_gain(g_vector: np.ndarray) -> np.ndarray:
    S = len(g_vector); g_out = g_vector.copy().astype(np.float32)
    for l in range(S): g_out[l] *= (100.0 + (1.0/20.0)*(l+1)*np.sqrt(l+1))
    return g_out

def reconstruct_cgnr(H: np.ndarray, g: np.ndarray, max_iterations: int, tol=5e-3, min_iterations=10, lambda_reg: float = 0.0, logger=None) -> tuple:
    m, n = H.shape
    # 1. Normalizacao automatica
    H_norm = np.linalg.norm(H)
    g_norm = np.linalg.norm(g)
    if H_norm == 0 or g_norm == 0:
        raise ValueError("H ou g sao nulos, impossivel normalizar.")
    Hn = H / H_norm
    gn = g.reshape(-1, 1) / g_norm

    # 1b. Regularizacao de Tikhonov (adiciona linhas a matriz e zeros ao vetor)
    if lambda_reg > 0:
        Hn = np.vstack([Hn, np.sqrt(lambda_reg) * np.eye(n)])
        gn = np.vstack([gn, np.zeros((n, 1))])

    # 2. Tolerancia adaptativa (relativa a escala dos dados)
    tol = max(tol, 1e-12)
    tol = tol * np.linalg.norm(gn)

    # 3. Inicializacao
    f = np.zeros((n, 1))
    r = gn - Hn @ f
    z = Hn.T @ r
    p = z.copy()
    initial_residual_norm = np.linalg.norm(r)
    number_iterations = 0
    min_div = 1e-12

    if logger is not None:
        logger.info(f"CGNR normalizado: H_norm={H_norm:.3e}, g_norm={g_norm:.3e}, tol={tol:.3e}, lambda={lambda_reg:.2e}")

    while number_iterations < max_iterations:
        w = Hn @ p
        z_dot = float(z.T @ z)
        w_dot = float(w.T @ w)

        alpha = z_dot / (w_dot + min_div)
        f_new = f + alpha * p
        r_new = r - alpha * w
        z_new = Hn.T @ r_new
        z_new_dot = float(z_new.T @ z_new)
        beta = z_new_dot / (z_dot + min_div)
        p_new = z_new + beta * p

        current_residual_norm = np.linalg.norm(r_new)
        relative_error = current_residual_norm / (initial_residual_norm + min_div)

        if logger is not None:
            logger.info(
                f"Iteracao {number_iterations + 1}: erro relativo = {relative_error:.6e}, "
                f"residuo = {current_residual_norm:.3e}"
            )
        if relative_error < tol:
            if logger is not None:
                logger.info(f"Convergiu com erro relativo {relative_error:.2e} < {tol:.2e}")
            f = f_new
            break

        f, r, z, p = f_new, r_new, z_new, p_new
        number_iterations += 1

    # 4. Desnormalizacao do resultado
    f = f * (g_norm / H_norm)

    # 5. Regularizacao e nao-negatividade (mantem para compatibilidade)
    f = np.maximum(f, 0)

    # 6. Erro final na escala original
    final_residual = g.reshape(-1, 1) - H @ f.reshape(-1, 1)
    final_error = np.linalg.norm(final_residual) / (np.linalg.norm(g) + min_div)

    return f, number_iterations, final_error

def reconstruct_cgne(H: np.ndarray, g: np.ndarray, max_iterations: int, tol=1e-6, min_iterations=10, reg_factor: float = 0.0, logger=None) -> tuple[np.ndarray, int, float]:
    N = H.shape[1]
    f = np.zeros((N, 1), dtype=np.float32)
    g = g.reshape(-1, 1)
    r = g - H @ f
    p = H.T @ r
    initial_residual_norm = np.linalg.norm(r)
    final_iterations = 0
    min_div = 1e-12

    for i in range(1, max_iterations + 1):
        Hp = H @ p
        alpha_num = float(r.T @ r)
        alpha_den = float(Hp.T @ Hp) + min_div
        if alpha_den < min_div:
            break
        alpha = alpha_num / alpha_den
        f_new = f + alpha * p
        r_new = r - alpha * (H @ p)
        beta_num = float(r_new.T @ r_new)
        beta_den = float(r.T @ r) + min_div
        beta = beta_num / beta_den
        p_new = H.T @ r_new + beta * p

        current_residual_norm = np.linalg.norm(r_new)
        relative_error = current_residual_norm / (initial_residual_norm + min_div)

        if logger is not None:
            logger.info(f"Iteracao {i}: erro relativo = {relative_error:.6f}")

        f, r, p = f_new, r_new, p_new
        final_iterations = i

        if i > min_iterations and relative_error < tol:
            if logger is not None:
                logger.info(f"Convergiu com erro relativo {relative_error:.2e} < {tol:.2e}")
            break

    # Regularizacao e nao-negatividade
    if reg_factor > 0.0:
        f = np.maximum(f - reg_factor, 0)
    else:
        f = np.maximum(f, 0)

    final_error = np.linalg.norm(g - H @ f) / (np.linalg.norm(g) + min_div)
    return f.flatten(), final_iterations, final_error


# Semáforos para concorrência
semaforo_clientes = threading.Semaphore(2)
semaforo_processos = threading.Semaphore(5)

# Cache de modelos
modelos = {}

app = Flask(__name__)


@app.route('/reconstruct', methods=['POST'])
def handle_reconstruction():
    """
    Novo endpoint: recebe caminho dos arquivos, lê do disco, processa e retorna PNG com metadados nos headers.
    Espera JSON: {"user":..., "algorithm":..., "model_path":..., "signal_path":..., ...}
    """
    with semaforo_clientes:
        data = request.get_json()
        if not all(k in data for k in ['user', 'algorithm', 'model_path', 'signal_path']):
            return jsonify({"error": "Requisicao incompleta. Esperado: user, algorithm, model_path, signal_path"}), 400

        user = data['user']
        algorithm = data['algorithm']
        model_path = data['model_path']
        signal_path = data['signal_path']
        use_regularization = data.get("regularization", False)
        reg_factor = data.get("reg_factor", 0.1)

        logging.info(f"Req de '{user}' para '{algorithm}' (Reg: {use_regularization}, Fator: {reg_factor})")

        start_time = time.time()
        start_dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with semaforo_processos:
            try:
                # Carrega modelo (matriz H) do disco, com cache
                if model_path in modelos:
                    H_matrix = modelos[model_path]
                else:
                    H_matrix = np.loadtxt(model_path, delimiter=',', dtype=np.float32)
                    modelos[model_path] = H_matrix

                g_vector = np.loadtxt(signal_path, delimiter=',', dtype=np.float32)
                g_processed = apply_signal_gain(g_vector)

                if use_regularization:
                    lambda_reg = np.max(np.abs(H_matrix.T @ g_vector)) * reg_factor
                    logging.info(f"Regularizacao ativada com lambda (fator {reg_factor}) = {lambda_reg:.4f}")
                    H_to_use = np.vstack([H_matrix, lambda_reg * np.identity(H_matrix.shape[1], dtype=np.float32)])
                    g_to_use = np.hstack([g_processed, np.zeros(H_matrix.shape[1], dtype=np.float32)])
                else:
                    logging.info("Regularizacao nao solicitada.")
                    H_to_use, g_to_use = H_matrix, g_processed

                # Reconstrução
                if algorithm.upper() == 'CGNR':
                    f, iters, final_error = reconstruct_cgnr(H_to_use, g_to_use, 100)
                elif algorithm.upper() == 'CGNE':
                    f, iters, final_error = reconstruct_cgne(H_to_use, g_to_use, 100)
                else:
                    return jsonify({"error": f"Algoritmo '{algorithm}' não suportado."}), 400

                # Normaliza para 0–255
                f = f.flatten()
                f_min, f_max = f.min(), f.max()
                if f_max != f_min:
                    f_norm = (f - f_min) / (f_max - f_min) * 255
                else:
                    f_norm = np.full_like(f, 128)

                lado = int(np.sqrt(len(f_norm)))
                imagem_array = f_norm[:lado*lado].reshape((lado, lado), order='F')
                imagem_array = np.clip(imagem_array, 0, 255)
                imagem = Image.fromarray(imagem_array.astype('uint8'))

                img_bytes = io.BytesIO()
                imagem.save(img_bytes, format='PNG')
                img_bytes.seek(0)

                end_time = time.time()
                end_dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                mem = psutil.virtual_memory()
                cpu = psutil.cpu_percent(interval=1)

                response = make_response(send_file(img_bytes, mimetype='image/png', download_name='reconstruida.png'))
                response.headers['X-Usuario'] = user
                response.headers['X-Algoritmo'] = algorithm
                response.headers['X-Inicio'] = start_dt
                response.headers['X-Fim'] = end_dt
                response.headers['X-Tamanho'] = f"{lado}x{lado}"
                response.headers['X-Iteracoes'] = str(iters)
                response.headers['X-Tempo'] = str(end_time - start_time)
                response.headers['X-Cpu'] = str(cpu)
                response.headers['X-Mem'] = str(mem.percent)
                return response
            except Exception as e:
                logging.error(f"Erro ao executar reconstrução: {e}")
                return jsonify({'error': str(e)}), 500


# endpoint para verificar se o servidor ligou
@app.route('/ping', methods=["GET"])
def ping():
    return 'OK', 200

if __name__ == '__main__':
    logging.info("Servidor de reconstrucao v2.6 (Final) iniciado.")
    print("Endpoints disponíveis:")
    print("  POST /reconstruct - Processa imagem")
    print("  GET /ping - Testa servidor")
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)