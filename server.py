# -----------------------------------------------------------------------------
# server.py - Servidor de Reconstrução de Imagem (v2.6)
#
# Melhorias:
# - Aceita o fator de regularização dinamicamente do cliente para ajuste fino.
# -----------------------------------------------------------------------------
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
import logging
from numba import jit

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler("server.log"), logging.StreamHandler()])

@jit(nopython=True)
def apply_signal_gain(g_vector: np.ndarray) -> np.ndarray:
    S = len(g_vector); g_out = g_vector.copy().astype(np.float32)
    for l in range(S): g_out[l] *= (100.0 + (1.0/20.0)*(l+1)*np.sqrt(l+1))
    return g_out

@jit(nopython=True)
def reconstruct_cgnr(H: np.ndarray, g: np.ndarray, max_iterations: int, tol=1e-6, min_iterations=10) -> tuple[np.ndarray, int]:
    N = H.shape[1]; f = np.zeros(N, dtype=np.float32); r = g - H @ f; z = H.T @ r; p = z
    z_norm_sq = np.dot(z, z); norm_r_old = np.linalg.norm(r); final_iterations = 0
    for i in range(1, max_iterations + 1):
        final_iterations = i; w = H @ p; w_norm_sq = np.dot(w, w)
        if w_norm_sq < 1e-12: break
        alpha = z_norm_sq / w_norm_sq; f += alpha * p; r -= alpha * w
        norm_r_new = np.linalg.norm(r)
        if i > min_iterations and np.abs(norm_r_old - norm_r_new) < tol: break
        norm_r_old = norm_r_new; z_next = H.T @ r; z_next_norm_sq = np.dot(z_next, z_next)
        beta = z_next_norm_sq / z_norm_sq; p = z_next + beta * p; z_norm_sq = z_next_norm_sq
    return f, final_iterations

@jit(nopython=True)
def reconstruct_cgne(H: np.ndarray, g: np.ndarray, max_iterations: int, tol=1e-6, min_iterations=10) -> tuple[np.ndarray, int]:
    N = H.shape[1]; f = np.zeros(N, dtype=np.float32); r = g - H @ f; p = H.T @ r; final_iterations = 0
    for i in range(1, max_iterations + 1):
        final_iterations = i; r_norm_sq = np.dot(r, r); p_norm_sq = np.dot(p, p)
        if p_norm_sq < 1e-12: break
        alpha = r_norm_sq / p_norm_sq; f += alpha * p; r_next = r - alpha * (H @ p)
        if i > min_iterations and np.abs(np.linalg.norm(r) - np.linalg.norm(r_next)) < tol: r = r_next; break
        beta = np.dot(r_next, r_next) / r_norm_sq; p = (H.T @ r_next) + beta * p; r = r_next
    return f, final_iterations

app = Flask(__name__)

@app.route('/reconstruct', methods=['POST'])
def handle_reconstruction():
    data = request.get_json()
    if not all(k in data for k in ['user', 'algorithm', 'H', 'g']):
        return jsonify({"error": "Requisição incompleta."}), 400

    user, algorithm = data['user'], data['algorithm']
    use_regularization = data.get("regularization", False)
    # Recebe o fator do cliente, com 0.1 como valor padrão de segurança
    reg_factor = data.get("reg_factor", 0.1)

    logging.info(f"Req de '{user}' para '{algorithm}' (Reg: {use_regularization}, Fator: {reg_factor})")

    H_matrix = np.array(data['H'], dtype=np.float32)
    g_vector = np.array(data['g'], dtype=np.float32)
    
    g_processed = apply_signal_gain(g_vector)
    
    if use_regularization:
        # Usa o fator recebido do cliente para calcular lambda
        lambda_reg = np.max(np.abs(H_matrix.T @ g_vector)) * reg_factor
        logging.info(f"Regularização ativada com λ (fator {reg_factor}) = {lambda_reg:.4f}")
        H_to_use = np.vstack([H_matrix, lambda_reg * np.identity(H_matrix.shape[1], dtype=np.float32)])
        g_to_use = np.hstack([g_processed, np.zeros(H_matrix.shape[1], dtype=np.float32)])
    else:
        logging.info("Regularização não solicitada.")
        H_to_use, g_to_use = H_matrix, g_processed

    start_time = datetime.now()
    if algorithm == 'CGNR':
        image, iters = reconstruct_cgnr(H_to_use, g_to_use, 100)
    elif algorithm == 'CGNE':
        image, iters = reconstruct_cgne(H_to_use, g_to_use, 100)
    else:
        return jsonify({"error": f"Algoritmo '{algorithm}' não suportado."}), 400
    
    end_time = datetime.now()
    logging.info(f"Concluído em {end_time - start_time}. Iters: {iters}.")
    
    response_data = { "metadata": { "user_id": user, "algorithm_used": algorithm, "start_time": start_time.isoformat(), "end_time": end_time.isoformat(), "duration_seconds": (end_time - start_time).total_seconds(), "image_size_pixels": len(image), "iterations_executed": iters, "regularization_used": use_regularization, "regularization_factor": reg_factor, "stopped_by_tolerance": iters < 100 }, "image_data": image.tolist() }
    return jsonify(response_data)

if __name__ == '__main__':
    logging.info("Servidor de reconstrução v2.6 (Final) iniciado.")
    app.run(host='0.0.0.0', port=5000, debug=False)