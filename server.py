import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, send_file, make_response
import logging
from numba import jit
import threading
import queue
import os
import io
from PIL import Image
import random
import psutil
import time
import gc
from algoritmos import *

app = Flask(__name__)
batch_status_dict = {}

# Fila de prioridade para jobs batch
batch_priority_queue = queue.PriorityQueue()

@app.route('/batch_reconstruct', methods=['POST'])
def batch_reconstruct():
    """
    Recebe uma lista de jobs para reconstrução e enfileira todos para processamento assíncrono.
    Cada job deve conter: user, algorithm, model_path, signal_path, regularization, reg_factor.
    O servidor responde imediatamente e processa cada job em background, salvando as imagens em 'outputs/'.
    """
    jobs = request.get_json()
    if not isinstance(jobs, list) or not jobs:
        return jsonify({"error": "Envie uma lista de jobs no corpo da requisição."}), 400

    os.makedirs("outputs", exist_ok=True)
    ids = []


    for job in jobs:
        # Gera um id único para cada job
        job_id = f"job_{int(time.time()*1000)}_{random.randint(1000,9999)}"
        ids.append(job_id)

        def process_job(job=job, job_id=job_id):
            try:
                user = job.get('user', 'anon')
                algorithm = job.get('algorithm', 'CGNE')
                model_path = job.get('model_path')
                signal_path = job.get('signal_path')
                use_regularization = job.get('regularization', False)
                reg_factor = job.get('reg_factor', 0.1)
                if not (model_path and signal_path):
                    logging.error(f"[BATCH] Job {job_id} faltando model_path ou signal_path.")
                    return
                if not os.path.isfile(model_path) or not os.path.isfile(signal_path):
                    logging.error(f"[BATCH] Job {job_id} arquivos não encontrados.")
                    return
                # Carrega modelo e sinal
                if model_path in modelos:
                    H_matrix = modelos[model_path]
                else:
                    H_matrix = np.loadtxt(model_path, delimiter=',', dtype=np.float32)
                    modelos[model_path] = H_matrix
                g_vector = np.loadtxt(signal_path, delimiter=',', dtype=np.float32)
                g_processed = apply_signal_gain(g_vector)
                if use_regularization:
                    lambda_reg = np.max(np.abs(H_matrix.T @ g_vector)) * reg_factor
                    H_to_use = np.vstack([H_matrix, lambda_reg * np.identity(H_matrix.shape[1], dtype=np.float32)])
                    g_to_use = np.hstack([g_processed, np.zeros(H_matrix.shape[1], dtype=np.float32)])
                else:
                    H_to_use, g_to_use = H_matrix, g_processed
                tol_requisito = 1e-4
                start_time = time.time()
                start_dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                if algorithm.upper() == 'CGNR':
                    f, iters, final_error = reconstruct_cgnr(H_to_use, g_to_use, 5, tol=tol_requisito)
                elif algorithm.upper() == 'CGNE':
                    f, iters, final_error = reconstruct_cgne(H_to_use, g_to_use, 5, tol=tol_requisito)
                else:
                    logging.error(f"[BATCH] Job {job_id} algoritmo não suportado.")
                    return
                end_time = time.time()
                end_dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f = f.flatten()
                f_min, f_max = f.min(), f.max()
                if f_max != f_min:
                    f_norm = (f - f_min) / (f_max - f_min) * 255
                else:
                    f_norm = np.full_like(f, 128)
                lado = int(np.sqrt(len(f_norm)))
                imagem_array = f_norm[:lado*lado].reshape((lado, lado), order='F')
                imagem_array = np.clip(imagem_array, 0, 255)
                # Salva usando matplotlib
                import matplotlib.pyplot as plt
                output_path = os.path.join("outputs", f"{job_id}_{user}_{algorithm}_{os.path.basename(model_path)}_{os.path.basename(signal_path)}.png")
                plt.figure(figsize=(6,6))
                plt.imshow(imagem_array, cmap="gray")
                plt.axis("off")
                plt.tight_layout(pad=0)
                plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
                plt.close()
                mem = psutil.virtual_memory()
                cpu = psutil.cpu_percent(interval=0.2)
                # Salva metadados em memória para consulta pelo cliente
                batch_status_dict[job_id] = {
                    "usuario": user,
                    "algoritmo": algorithm,
                    "inicio": start_dt,
                    "fim": end_dt,
                    "tamanho": f"{lado}x{lado}",
                    "iteracoes": iters,
                    "tempo": round(end_time - start_time, 2),
                    "cpu": round(cpu, 1),
                    "mem": round(mem.percent, 1),
                    "imagem": os.path.basename(output_path)
                }
                logging.info(f"[BATCH] Job {job_id} salvo em {output_path} e metadados em memória.")
            except Exception as e:
                logging.error(f"[BATCH] Erro no job {job_id}: {e}")

        # Define prioridade: menor valor = maior prioridade
        # matriz H-2 + CGNE = prioridade 0
        # matriz H-2 ou CGNE = prioridade 1
        # outros = prioridade 2
        model_name = os.path.basename(job.get('model_path', '')).lower()
        alg_name = job.get('algorithm', '').upper()
        if 'h-2' in model_name and alg_name == 'CGNE':
            priority = 0
        elif 'h-2' in model_name or alg_name == 'CGNE':
            priority = 1
        else:
            priority = 2
        batch_priority_queue.put((priority, time.time(), process_job, [], {}))

    # Retorna resposta imediatamente após enfileirar os jobs
    return jsonify({
        "status": "Jobs recebidos",
        "job_ids": ids,
        "msg": "As imagens serão processadas e salvas em 'outputs/'."
    }), 202
    
@app.route('/download_image', methods=['GET'])
def download_image():
    """
    Permite ao cliente baixar a imagem reconstruída de um job pelo job_id.
    O cliente deve consultar /batch_status até o job estar pronto, depois chamar este endpoint.
    Exemplo: GET /download_image?job_id=job_1234
    """
    job_id = request.args.get('job_id')
    if not job_id:
        return jsonify({"error": "Forneça o parâmetro job_id na query."}), 400
    job_info = batch_status_dict.get(job_id)
    if not job_info:
        return jsonify({"error": "Job não encontrado ou ainda não processado."}), 404
    image_name = job_info.get("imagem")
    if not image_name:
        return jsonify({"error": "Imagem não disponível para este job."}), 404
    image_path = os.path.join("outputs", image_name)
    if not os.path.isfile(image_path):
        return jsonify({"error": "Arquivo de imagem não encontrado."}), 404
    try:
        return send_file(image_path, mimetype='image/png', download_name=image_name)
    except Exception as e:
        return jsonify({"error": f"Erro ao enviar imagem: {str(e)}"}), 500

# Endpoint para consultar status dos jobs batch
@app.route('/batch_status', methods=['GET'])
def batch_status():
    job_ids = request.args.get('job_ids')
    if not job_ids:
        return jsonify({"error": "Forneça job_ids separados por vírgula."}), 400
    ids = [jid.strip() for jid in job_ids.split(',') if jid.strip()]
    result = {jid: batch_status_dict.get(jid, None) for jid in ids}
    return jsonify(result)

os.makedirs("log", exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(os.path.join("log", "server.log")), logging.StreamHandler()])

@jit(nopython=True)
def apply_signal_gain(g_vector: np.ndarray) -> np.ndarray:
    S = len(g_vector); g_out = g_vector.copy().astype(np.float32)
    for l in range(S): g_out[l] *= (100.0 + (1.0/20.0)*(l+1)*np.sqrt(l+1))
    return g_out

# Semáforos para concorrência
semaforo_clientes = threading.Semaphore(3)
semaforo_processos = threading.Semaphore(5)

# Cache de modelos
modelos = {}


# Funções para limites dinâmicos de uso de recursos
def get_dynamic_cpu_limit():
    # Limite: 80% dos núcleos lógicos
    n_cores = psutil.cpu_count(logical=True)
    return max(50.0, min(90.0, n_cores * 80.0 / n_cores))  # 80% (ajustável)

def get_dynamic_mem_limit():
    # Limite: 80% da RAM total, mas sempre deixa pelo menos 1GB livre
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024**3)
    if total_gb <= 2:
        return 70.0  # Em PCs com pouca RAM, seja mais conservador
    # 80% do total, mas nunca usar mais que total-1GB
    max_percent = 100.0 - (1.0 / total_gb) * 100.0
    return min(80.0, max_percent)


# Fila para requisições pendentes (reconstrução individual)
request_queue = queue.Queue()


# Função worker para processar a fila de prioridade dos jobs batch
def process_batch_priority_queue_worker():
    while True:
        item = batch_priority_queue.get()
        if item is None:
            break
        _, _, func, args, kwargs = item
        try:
            func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Erro no processamento da fila batch: {e}")
        batch_priority_queue.task_done()

# Função worker para processar a fila de requisições individuais
def process_queue_worker():
    while True:
        item = request_queue.get()
        if item is None:
            break
        func, args, kwargs = item
        try:
            func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Erro no processamento da fila: {e}")
        request_queue.task_done()

# Inicia os workers das filas
worker_thread = threading.Thread(target=process_queue_worker, daemon=True)
worker_thread.start()
batch_worker_thread = threading.Thread(target=process_batch_priority_queue_worker, daemon=True)
batch_worker_thread.start()

@app.route('/reconstruct', methods=['POST'])
def handle_reconstruction():
    """
    Endpoint robusto para reconstrução de imagem.
    Recebe caminhos dos arquivos, valida, lê, processa e retorna PNG com metadados.
    """
    # Checagem dinâmica de recursos antes de aceitar a requisição
    def process_request():
        with semaforo_clientes:
            data = request.get_json()
            if not all(k in data for k in ['user', 'algorithm', 'model_path', 'signal_path']):
                return jsonify({"error": "Requisição incompleta. Esperado: user, algorithm, model_path, signal_path"}), 400

            user = data['user']
            algorithm = data['algorithm']
            model_path = data['model_path']
            signal_path = data['signal_path']
            use_regularization = data.get("regularization", False)
            reg_factor = data.get("reg_factor", 0.1)

            logging.info(f"[INICIO] Usuário: {user} | Algoritmo: {algorithm} | Reg: {use_regularization} | Fator: {reg_factor}")
            start_time = time.time()
            start_dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Validação de existência dos arquivos
            if not os.path.isfile(model_path):
                logging.error(f"Arquivo de modelo não encontrado: {model_path}")
                return jsonify({"error": f"Arquivo de modelo não encontrado: {model_path}"}), 400
            if not os.path.isfile(signal_path):
                logging.error(f"Arquivo de sinal não encontrado: {signal_path}")
                return jsonify({"error": f"Arquivo de sinal não encontrado: {signal_path}"}), 400

            with semaforo_processos:
                try:
                    # Carrega modelo (matriz H) do disco, com cache
                    if model_path in modelos:
                        H_matrix = modelos[model_path]
                        logging.info(f"Matriz H carregada do cache: {model_path}")
                    else:
                        logging.info(f"Lendo matriz H do disco: {model_path}")
                        H_matrix = np.loadtxt(model_path, delimiter=',', dtype=np.float32)
                        modelos[model_path] = H_matrix
                    logging.info(f"Matriz H: shape={H_matrix.shape}, dtype={H_matrix.dtype}")

                    logging.info(f"Lendo vetor g do disco: {signal_path}")
                    g_vector = np.loadtxt(signal_path, delimiter=',', dtype=np.float32)
                    logging.info(f"Vetor g: shape={g_vector.shape}, dtype={g_vector.dtype}")

                    g_processed = apply_signal_gain(g_vector)

                    if use_regularization:
                        lambda_reg = np.max(np.abs(H_matrix.T @ g_vector)) * reg_factor
                        logging.info(f"Regularizacao ativada com lambda (fator {reg_factor}) = {lambda_reg:.4f}")
                        H_to_use = np.vstack([H_matrix, lambda_reg * np.identity(H_matrix.shape[1], dtype=np.float32)])
                        g_to_use = np.hstack([g_processed, np.zeros(H_matrix.shape[1], dtype=np.float32)])
                    else:
                        logging.info("Regularizacao nao solicitada.")
                        H_to_use, g_to_use = H_matrix, g_processed

                    # Reconstrução com tolerância conforme requisito (1e-4)
                    tol_requisito = 1e-4
                    logging.info(f"Iniciando reconstrução ({algorithm.upper()}) com tolerância {tol_requisito}...")
                    if algorithm.upper() == 'CGNR':
                        f, iters, final_error = reconstruct_cgnr(H_to_use, g_to_use, 5, tol=tol_requisito)
                    elif algorithm.upper() == 'CGNE':
                        f, iters, final_error = reconstruct_cgne(H_to_use, g_to_use, 5, tol=tol_requisito)
                    else:
                        return jsonify({"error": f"Algoritmo '{algorithm}' não suportado."}), 400
                    logging.info(f"Reconstrução finalizada. Iterações: {iters} | Erro final: {final_error:.3e}")

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

                    # Libera memória de variáveis grandes
                    del H_matrix, g_vector, g_processed, H_to_use, g_to_use, f, f_norm, imagem_array, imagem
                    gc.collect()

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
                    logging.info(f"[FIM] Usuário: {user} | Tempo total: {end_time - start_time:.1f}s | Iterações: {iters}")
                    return response
                except Exception as e:
                    logging.error(f"Erro ao executar reconstrução: {e}")
                    return jsonify({'error': f'Erro interno: {str(e)}'}), 500

    # Checagem dinâmica de recursos antes de aceitar a requisição

    cpu_limit = get_dynamic_cpu_limit()
    mem_limit = get_dynamic_mem_limit()
    cpu_percent = psutil.cpu_percent(interval=0.5)
    mem_percent = psutil.virtual_memory().percent
    if cpu_percent > cpu_limit or mem_percent > mem_limit:
        logging.warning(f"Recursos insuficientes: CPU={cpu_percent:.1f}%, MEM={mem_percent:.1f}% (limites dinâmicos: CPU={cpu_limit:.1f}%, MEM={mem_limit:.1f}%) - Requisição será enfileirada.")
        # Enfileira a requisição para ser processada depois
        def delayed_response():
            # Aguarda até recursos ficarem disponíveis
            while True:
                cpu = psutil.cpu_percent(interval=0.5)
                mem = psutil.virtual_memory().percent
                if cpu <= cpu_limit and mem <= mem_limit:
                    break
                time.sleep(1)
            # Processa normalmente
            with app.app_context():
                return process_request()
        # Adiciona na fila
        request_queue.put((process_request, [], {}))
        return jsonify({"status": "Aguardando na fila. Sua requisição será processada assim que possível."}), 202
    else:
        return process_request()


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