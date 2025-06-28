# -----------------------------------------------------------------------------
# server.py - Servidor de Reconstrução de Imagem (v2.6)
# -----------------------------------------------------------------------------
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
import logging
from numba import jit

import os
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

app = Flask(__name__)

@app.route('/reconstruct', methods=['POST'])
def handle_reconstruction():
    data = request.get_json()
    if not all(k in data for k in ['user', 'algorithm', 'H', 'g']):
        return jsonify({"error": "Requisicao incompleta."}), 400

    user, algorithm = data['user'], data['algorithm']
    use_regularization = data.get("regularization", False)
    # Recebe o fator do cliente, com 0.1 como valor padrao de seguranca
    reg_factor = data.get("reg_factor", 0.1)

    logging.info(f"Req de '{user}' para '{algorithm}' (Reg: {use_regularization}, Fator: {reg_factor})")

    H_matrix = np.array(data['H'], dtype=np.float32)
    g_vector = np.array(data['g'], dtype=np.float32)
    
    g_processed = apply_signal_gain(g_vector)
    
    if use_regularization:
        # Usa o fator recebido do cliente para calcular lambda
        lambda_reg = np.max(np.abs(H_matrix.T @ g_vector)) * reg_factor
        logging.info(f"Regularizacao ativada com lambda (fator {reg_factor}) = {lambda_reg:.4f}")
        H_to_use = np.vstack([H_matrix, lambda_reg * np.identity(H_matrix.shape[1], dtype=np.float32)])
        g_to_use = np.hstack([g_processed, np.zeros(H_matrix.shape[1], dtype=np.float32)])
    else:
        logging.info("Regularizacao nao solicitada.")
        H_to_use, g_to_use = H_matrix, g_processed

    start_time = datetime.now()
    image = None
    iters = None
    final_error = None
    try:
        if algorithm == 'CGNR':
            result = reconstruct_cgnr(H_to_use, g_to_use, 100)
            if isinstance(result, tuple) and len(result) == 2:
                image, iters = result
                final_error = None
            elif isinstance(result, tuple) and len(result) == 3:
                image, iters, final_error = result
                logging.warning("reconstruct_cgnr retornou 3 valores!")
            else:
                raise ValueError("Retorno inesperado de reconstruct_cgnr: " + str(result))
        elif algorithm == 'CGNE':
            result = reconstruct_cgne(H_to_use, g_to_use, 100)
            if isinstance(result, tuple) and len(result) == 3:
                image, iters, final_error = result
            elif isinstance(result, tuple) and len(result) == 2:
                image, iters = result
                final_error = None
                logging.warning("reconstruct_cgne retornou apenas 2 valores!")
            else:
                raise ValueError("Retorno inesperado de reconstruct_cgne: " + str(result))
            logging.info(f"CGNE: iters={iters}, erro_final={final_error}")
        else:
            return jsonify({"error": f"Algoritmo '{algorithm}' não suportado."}), 400
    except Exception as e:
        logging.error(f"Erro ao executar reconstrução: {e}")
        return jsonify({"error": f"Erro interno na reconstrução: {e}"}), 500

    end_time = datetime.now()
    logging.info(f"Concluido em {end_time - start_time}. Iters: {iters}.")

    response_data = { "metadata": { "user_id": user, "algorithm_used": algorithm, "start_time": start_time.isoformat(), "end_time": end_time.isoformat(), "duration_seconds": (end_time - start_time).total_seconds(), "image_size_pixels": len(image), "iterations_executed": iters, "regularization_used": use_regularization, "regularization_factor": reg_factor, "stopped_by_tolerance": iters < 100 }, "image_data": image.tolist() }
    return jsonify(response_data)

if __name__ == '__main__':
    logging.info("Servidor de reconstrucao v2.6 (Final) iniciado.")
    app.run(host='0.0.0.0', port=5000, debug=False)