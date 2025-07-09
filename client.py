
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
import numpy as np
import requests
import logging
import threading
from datetime import datetime
from matplotlib import pyplot as plt
import os
import random
import time

# Listas para modo automático
USUARIOS = ['Alice', 'José', 'Carol', 'Daniel','Murilo','Maria','Ana','Leonardo','Eduarda','Lucas']
MODELOS = ['data/H-1.csv', 'data/H-2.csv']
SINAIS60 = ['data/G-1.csv', 'data/G-2.csv', 'data/A-60x60-1.csv']
SINAIS30 = ['data/g-30x30-1.csv','data/g-30x30-2.csv','data/A-30x30-1.csv']
TODOS_SINAIS = SINAIS60 + SINAIS30

os.makedirs("log", exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.FileHandler(os.path.join("log", "client.log"))])

class ReconstructionClientApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Cliente de Reconstrução de Imagem v2.6")
        self.geometry("700x700")

        # Variáveis de estado
        self.last_result_data, self.last_algorithm_used = None, None
        self.path_h, self.path_g = tk.StringVar(), tk.StringVar()
        self.username = tk.StringVar(value="user_default")
        self.algorithm = tk.StringVar(value="CGNE") # algoritmo padrão
        self.use_regularization = tk.BooleanVar(value=False)
        self.use_log_scale = tk.BooleanVar(value=False)
        self.reg_factor = tk.DoubleVar(value=0.1) # regularização
        self.zoom_factor = tk.IntVar(value=1) # zoom
        # Construção da Interface
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill="both", expand=True)
        input_frame = ttk.LabelFrame(main_frame, text="Configuração da Reconstrução", padding="10")
        input_frame.pack(fill="x")

        ttk.Label(input_frame, text="Execuções automáticas:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.n_execucoes = tk.IntVar(value=5)
        ttk.Spinbox(input_frame, from_=1, to=100, textvariable=self.n_execucoes, width=6).grid(row=0, column=1, sticky="w", padx=5)

        ttk.Label(input_frame, text="Intervalo (s):").grid(row=0, column=2, sticky="w", padx=5)
        self.intervalo_min = tk.DoubleVar(value=0.5)
        self.intervalo_max = tk.DoubleVar(value=2.0)
        ttk.Entry(input_frame, textvariable=self.intervalo_min, width=5).grid(row=0, column=3, sticky="w")
        ttk.Label(input_frame, text="a").grid(row=0, column=4, sticky="w")
        ttk.Entry(input_frame, textvariable=self.intervalo_max, width=5).grid(row=0, column=5, sticky="w")

        action_frame = ttk.Frame(main_frame); action_frame.pack(pady=10)
        self.run_button = ttk.Button(action_frame, text="Iniciar Execução Automática", command=self.start_auto_thread)
        self.run_button.pack(side="left", padx=5)
        self.plot_button = ttk.Button(action_frame, text="Plotar Resultado", command=self.plot_result, state="disabled")
        self.plot_button.pack(side="left", padx=5)


        log_frame = ttk.LabelFrame(main_frame, text="Log de Execução e Resultados", padding="10")
        log_frame.pack(fill="both", expand=True)
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, state="disabled")
        self.log_text.pack(fill="both", expand=True)

    def start_auto_thread(self):
        self.run_button.config(state="disabled")
        self.plot_button.config(state="disabled")
        self.log("Iniciando execuções automáticas...")
        thread = threading.Thread(target=self.run_auto_mode)
        thread.start()

    def run_auto_mode(self):
        import time
        n = self.n_execucoes.get()
        jobs = []
        for i in range(n):
            user = random.choice(USUARIOS)
            modelo = random.choice(MODELOS)
            # Seleciona sinal compatível com o modelo
            if modelo.lower().endswith('h-1.csv'):
                sinal = random.choice(SINAIS60)
            else:
                sinal = random.choice(SINAIS30)
            algoritmo = random.choice(["CGNR", "CGNE"])
            job = {
                "user": user,
                "algorithm": algoritmo,
                "model_path": modelo,
                "signal_path": sinal,
                "regularization": self.use_regularization.get(),
                "reg_factor": self.reg_factor.get()
            }
            jobs.append(job)
            self.log(f"[{i+1}/{n}] (batch) Usuário: {user} | Modelo: {modelo} | Sinal: {sinal} | Algoritmo: {algoritmo}")
        # Envia todos os jobs de uma vez
        try:
            response = requests.post("http://127.0.0.1:5000/batch_reconstruct", json=jobs, timeout=60)
            if response.status_code == 202:
                self.log("Jobs enviados para o servidor! O processamento será feito em background.")
                # Recupera os job_ids retornados
                job_ids = response.json().get("job_ids", [])
                if job_ids:
                    self.log("Aguardando resultados dos jobs...")
                    self.poll_batch_status(job_ids)
            else:
                self.log(f"[ERRO] Falha ao enviar batch: {response.text}")
        except Exception as e:
            self.log(f"[ERRO] Falha ao enviar batch: {e}")
        self.run_button.config(state="normal")
        self.log("Execução automática finalizada.")

    def poll_batch_status(self, job_ids, interval=2, max_wait=300):
        """
        Consulta periodicamente o status dos jobs batch, exibe os resultados no log e salva nos relatórios.
        """
        import time
        import requests
        start_time = time.time()
        jobs_pendentes = set(job_ids)
        jobs_exibidos = set()
        while jobs_pendentes and (time.time() - start_time < max_wait):
            try:
                ids_str = ','.join(jobs_pendentes)
                url = f"http://127.0.0.1:5000/batch_status?job_ids={ids_str}"
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    status_dict = resp.json()
                    for jid, info in status_dict.items():
                        if info and jid not in jobs_exibidos:
                            self.log(f"\n--- RESULTADO BATCH {jid} ---")
                            for k, v in info.items():
                                self.log(f"{k:<12}: {v}")
                            self.log(f"Diretório da imagem: outputs/{info.get('imagem','')}\n")
                            # Salva nos relatórios
                            try:
                                # relatorio_imagens.txt (com todos os requisitos)
                                with open("relatorio_imagens.txt", "a", encoding="utf-8") as fimg:
                                    fimg.write(
                                        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                                        f"Imagem: outputs/{info.get('imagem','')} | "
                                        f"Usuario: {info.get('usuario','')} | "
                                        f"Algoritmo: {info.get('algoritmo','')} | "
                                        f"Inicio: {info.get('inicio','')} | "
                                        f"Fim: {info.get('fim','')} | "
                                        f"Tamanho: {info.get('tamanho','')} | "
                                        f"Iteracoes: {info.get('iteracoes','')}\n"
                                    )
                                # relatorio_desempenho.txt
                                with open("relatorio_desempenho.txt", "a", encoding="utf-8") as fdes:
                                    fdes.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Usuario: {info.get('usuario','')} | Algoritmo: {info.get('algoritmo','')} | CPU: {info.get('cpu','')}% | MEM: {info.get('mem','')}% | Tempo: {info.get('tempo','')}s\n")
                            except Exception as e:
                                self.log(f"[ERRO] Falha ao salvar relatórios batch: {e}")
                            jobs_exibidos.add(jid)
                    jobs_pendentes = jobs_pendentes - jobs_exibidos
                else:
                    self.log(f"[ERRO] Falha ao consultar status dos jobs: {resp.text}")
            except Exception as e:
                self.log(f"[ERRO] Consulta batch_status: {e}")
            self.update()
            time.sleep(interval)
        if jobs_pendentes:
            self.log(f"[AVISO] Timeout aguardando jobs: {', '.join(jobs_pendentes)}")
        else:
            self.log("Todos os resultados dos jobs batch foram exibidos.")

    def log(self, message):
        logging.info(message); self.log_text.config(state="normal")
        self.log_text.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
        self.log_text.config(state="disabled"); self.log_text.see(tk.END)

    def select_h_file(self):
        path = filedialog.askopenfilename(title="Selecione a Matriz H", filetypes=[("CSV", "*.csv")])
        if path: self.path_h.set(path); self.log(f"Arquivo H selecionado: {path}")

    def select_g_file(self):
        path = filedialog.askopenfilename(title="Selecione o Vetor G", filetypes=[("CSV", "*.csv")])
        if path: self.path_g.set(path); self.log(f"Arquivo G selecionado: {path}")

    def start_reconstruction_thread(self):
        self.run_button.config(state="disabled"); self.plot_button.config(state="disabled")
        self.log("Iniciando processo de reconstrução..."); thread = threading.Thread(target=self.run_reconstruction); thread.start()
        
    def run_reconstruction(self, auto=False):
        try:
            user, algo, path_h, path_g = self.username.get(), self.algorithm.get(), self.path_h.get(), self.path_g.get()
            if not all([user, algo, path_h, path_g]):
                self.log("ERRO: Todos os campos devem ser preenchidos.")
                return

            payload = {
                "user": user,
                "algorithm": algo,
                "model_path": path_h,
                "signal_path": path_g,
                "regularization": self.use_regularization.get(),
                "reg_factor": self.reg_factor.get()
            }

            self.log(f"Enviando requisição (Alg: {algo}, Reg: {self.use_regularization.get()}, Fator: {self.reg_factor.get()})...")
            try:
                response = requests.post("http://127.0.0.1:5000/reconstruct", json=payload, timeout=3600)
                response.raise_for_status()
            except requests.exceptions.Timeout:
                self.log("[ERRO] Tempo limite excedido. O servidor demorou demais para responder. Tente aumentar o timeout ou otimize os arquivos.")
                self.run_button.config(state="normal")
                return
            except requests.exceptions.ConnectionError:
                self.log("[ERRO] Não foi possível conectar ao servidor. Verifique se o servidor está rodando.")
                self.run_button.config(state="normal")
                return

            # Salva imagem recebida e aplica zoom se necessário
            output_dir = "outputs"
            os.makedirs(output_dir, exist_ok=True)
            nome_usuario = user.replace(' ', '_')
            nome_alg = algo.upper()
            output_filename = os.path.join(output_dir, f"resultado_{nome_usuario}_{nome_alg}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

            # Salva temporariamente
            temp_filename = output_filename + ".tmp"
            with open(temp_filename, 'wb') as f:
                f.write(response.content)

            # Aplica zoom se necessário
            zoom = self.zoom_factor.get()
            try:
                from PIL import Image
                img = Image.open(temp_filename)
                if zoom > 1:
                    w, h = img.size
                    img = img.resize((w*zoom, h*zoom), resample=Image.NEAREST)
                img.save(output_filename)
                os.remove(temp_filename)
            except Exception as e:
                os.rename(temp_filename, output_filename)
                self.log(f"[AVISO] Não foi possível aplicar zoom: {e}")

            self.log(f"SUCESSO: Imagem salva em '{output_filename}' (zoom x{zoom})")

            # Extrai metadados dos headers
            metadados = {}
            metadados_legiveis = []
            self.log("\n--- METADADOS DA RECONSTRUÇÃO ---")
            for key in [
                'X-Usuario', 'X-Algoritmo', 'X-Inicio', 'X-Fim', 'X-Tamanho',
                'X-Iteracoes', 'X-Tempo', 'X-Cpu', 'X-Mem']:
                value = response.headers.get(key, '')
                if value:
                    self.log(f"{key[2:]:<20}: {value}")
                    metadados[key] = value
                    metadados_legiveis.append(f"{key[2:]:<12}: {value}")
            self.log("----------------------------------\n")

            # Exibe relatório em popup
            if metadados_legiveis:
                try:
                    import tkinter.messagebox as msgbox
                    msgbox.showinfo("Relatório da Reconstrução",
                        "\n".join(metadados_legiveis), parent=self)
                except Exception as e:
                    self.log(f"[ERRO] Falha ao exibir relatório popup: {e}")

            # Salva relatórios na pasta relatorios/

            try:
                # Relatório de desempenho (CPU/MEM/TEMPO) -> relatorio_desempenho.txt
                with open("relatorio_desempenho.txt", "a", encoding="utf-8") as f:
                    f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Usuario: {metadados.get('X-Usuario','')} | Algoritmo: {metadados.get('X-Algoritmo','')} | CPU: {metadados.get('X-Cpu','')}% | MEM: {metadados.get('X-Mem','')}% | Tempo: {metadados.get('X-Tempo','')}s\n")

                # Relatório de imagens reconstruídas -> relatorio_imagens.txt
                # Inclui todos os dados obrigatórios
                with open("relatorio_imagens.txt", "a", encoding="utf-8") as f:
                    f.write(
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                        f"Imagem: {output_filename} | "
                        f"Usuario: {metadados.get('X-Usuario','')} | "
                        f"Algoritmo: {metadados.get('X-Algoritmo','')} | "
                        f"Inicio: {metadados.get('X-Inicio','')} | "
                        f"Fim: {metadados.get('X-Fim','')} | "
                        f"Tamanho: {metadados.get('X-Tamanho','')} | "
                        f"Iteracoes: {metadados.get('X-Iteracoes','')}\n"
                    )
            except Exception as e:
                self.log(f"[ERRO] Falha ao salvar relatórios: {e}")

            self.last_result_data = None
            self.last_algorithm_used = algo
            self.plot_button.config(state="disabled")
        except requests.exceptions.RequestException as e:
            self.log(f"ERRO DE COMUNICAÇÃO: {e}")
        except Exception as e:
            self.log(f"ERRO INESPERADO: {e}")
        finally:
            if not auto:
                self.run_button.config(state="normal")



    def process_image_for_display(self, data_vector):
        if self.use_log_scale.get(): self.log("Aplicando escala logarítmica."); return np.log1p(np.abs(data_vector))
        return data_vector

    def save_result_as_image(self) -> str:
        import os
        if self.last_result_data is None:
            return ""
        try:
            display_data = self.process_image_for_display(self.last_result_data)
            image_2d, dim = self.reshape_image_data(display_data)
            if image_2d is None:
                return ""
            fig = plt.figure("Salvar Imagem")
            plt.imshow(image_2d, cmap='gray', aspect='auto', interpolation='nearest')
            plt.colorbar(label="Intensidade (Escala Aplicada)")
            plt.title(f"Imagem {dim}x{dim} ({self.last_algorithm_used})")
            output_dir = "outputs"
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"resultado_{self.username.get()}_{self.algorithm.get()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return filename
        except Exception as e:
            self.log(f"Erro ao salvar o grafico: {e}")
            return ""

    def plot_result(self):
        if self.last_result_data is None: return
        self.log("Gerando gráfico...");
        try:
            display_data = self.process_image_for_display(self.last_result_data)
            image_2d, dim = self.reshape_image_data(display_data)
            if image_2d is None: return
            plt.figure("Visualização da Imagem"); plt.imshow(image_2d, cmap='gray', aspect='auto', interpolation='nearest')
            plt.colorbar(label="Intensidade (Escala Aplicada)"); plt.title(f"Imagem {dim}x{dim} ({self.last_algorithm_used})")
            plt.show()
        except Exception as e: self.log(f"Erro ao gerar o gráfico: {e}")

    def reshape_image_data(self, data_vector) -> tuple:
        n_pixels = len(data_vector); dim = int(np.sqrt(n_pixels))
        if dim * dim != n_pixels: self.log(f"ERRO: Resultado com {n_pixels} pixels não é quadrado."); return None, None
        return data_vector.reshape((dim, dim)), dim

if __name__ == "__main__":
    app = ReconstructionClientApp()
    app.mainloop()