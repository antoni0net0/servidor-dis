import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
import numpy as np
import requests
import logging
import threading
from datetime import datetime
from matplotlib import pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.FileHandler("client.log")])

class ReconstructionClientApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Cliente de Reconstrução de Imagem v2.6")
        self.geometry("700x680")

        # --- Variáveis de estado ---
        self.last_result_data, self.last_algorithm_used = None, None
        self.path_h, self.path_g = tk.StringVar(), tk.StringVar()
        self.username = tk.StringVar(value="user_default")
        self.algorithm = tk.StringVar(value="CGNE") # Padrão para o algoritmo que gerou a imagem de referência
        self.use_regularization = tk.BooleanVar(value=True)
        self.use_log_scale = tk.BooleanVar(value=True)
        self.reg_factor = tk.DoubleVar(value=0.1) # Novo: Variável para o fator de regularização

        # --- Construção da Interface ---
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill="both", expand=True)
        input_frame = ttk.LabelFrame(main_frame, text="Configuração da Reconstrução", padding="10")
        input_frame.pack(fill="x")

        ttk.Label(input_frame, text="Nome de Usuário:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(input_frame, textvariable=self.username).grid(row=0, column=1, columnspan=2, sticky="ew", padx=5)
        ttk.Label(input_frame, text="Algoritmo:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        ttk.Radiobutton(input_frame, text="CGNR", variable=self.algorithm, value="CGNR").grid(row=1, column=1, sticky="w", padx=5)
        ttk.Radiobutton(input_frame, text="CGNE", variable=self.algorithm, value="CGNE").grid(row=1, column=2, sticky="w", padx=5)
        ttk.Button(input_frame, text="Selecionar Matriz H (.csv)", command=self.select_h_file).grid(row=2, column=0, sticky="ew", padx=5, pady=2)
        ttk.Label(input_frame, textvariable=self.path_h, wraplength=450).grid(row=2, column=1, columnspan=2, sticky="w", padx=5)
        ttk.Button(input_frame, text="Selecionar Vetor G (.csv)", command=self.select_g_file).grid(row=3, column=0, sticky="ew", padx=5, pady=2)
        ttk.Label(input_frame, textvariable=self.path_g, wraplength=450).grid(row=3, column=1, columnspan=2, sticky="w", padx=5)
        
        # Frame para os controles de regularização
        reg_frame = ttk.Frame(input_frame)
        reg_frame.grid(row=4, column=0, columnspan=3, sticky="w")
        ttk.Checkbutton(reg_frame, text="Usar Regularização", variable=self.use_regularization).pack(side="left", padx=5, pady=5)
        ttk.Label(reg_frame, text="Fator:").pack(side="left", pady=5)
        ttk.Entry(reg_frame, textvariable=self.reg_factor, width=8).pack(side="left", pady=5)

        ttk.Checkbutton(input_frame, text="Usar Escala Logarítmica (Visualização)", variable=self.use_log_scale).grid(row=5, column=0, columnspan=3, sticky="w", padx=5, pady=5)

        action_frame = ttk.Frame(main_frame); action_frame.pack(pady=10)
        self.run_button = ttk.Button(action_frame, text="Iniciar Reconstrução", command=self.start_reconstruction_thread)
        self.run_button.pack(side="left", padx=5)
        self.plot_button = ttk.Button(action_frame, text="Plotar Resultado", command=self.plot_result, state="disabled")
        self.plot_button.pack(side="left", padx=5)

        log_frame = ttk.LabelFrame(main_frame, text="Log de Execução e Resultados", padding="10")
        log_frame.pack(fill="both", expand=True)
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, state="disabled")
        self.log_text.pack(fill="both", expand=True)

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
        
    def run_reconstruction(self):
        try:
            user, algo, path_h, path_g = self.username.get(), self.algorithm.get(), self.path_h.get(), self.path_g.get()
            if not all([user, algo, path_h, path_g]): self.log("ERRO: Todos os campos devem ser preenchidos."); return

            self.log("Carregando dados..."); H = np.loadtxt(path_h, delimiter=',', dtype=np.float32); g = np.loadtxt(path_g, delimiter=',', dtype=np.float32)
            
            payload = {"user": user, "algorithm": algo, "H": H.tolist(), "g": g.tolist(), "regularization": self.use_regularization.get(), "reg_factor": self.reg_factor.get()}

            self.log(f"Enviando requisição (Alg: {algo}, Reg: {self.use_regularization.get()}, Fator: {self.reg_factor.get()})...")
            response = requests.post("http://127.0.0.1:5000/reconstruct", json=payload, timeout=300)
            response.raise_for_status()
            
            results = response.json()
            self.last_result_data, self.last_algorithm_used = np.array(results['image_data']), results['metadata']['algorithm_used']
            self.log(f"Resposta recebida. Imagem contém {self.last_result_data.shape[0]} pixels.")
            
            self.log("\n--- METADADOS DA RECONSTRUÇÃO ---")
            for key, value in results['metadata'].items(): self.log(f"{str(key)+':':<35} {value}")
            self.log("----------------------------------\n")

            output_filename = self.save_result_as_image()
            self.log(f"SUCESSO: Gráfico salvo em '{output_filename}'")
            self.plot_button.config(state="normal")
        except requests.exceptions.RequestException as e: self.log(f"ERRO DE COMUNICAÇÃO: {e}")
        except Exception as e: self.log(f"ERRO INESPERADO: {e}")
        finally: self.run_button.config(state="normal")

    def process_image_for_display(self, data_vector):
        if self.use_log_scale.get(): self.log("Aplicando escala logarítmica."); return np.log1p(np.abs(data_vector))
        return data_vector

    def save_result_as_image(self) -> str:
        if self.last_result_data is None: return ""
        try:
            display_data = self.process_image_for_display(self.last_result_data)
            image_2d, dim = self.reshape_image_data(display_data)
            if image_2d is None: return ""
            fig = plt.figure("Salvar Imagem"); plt.imshow(image_2d, cmap='gray', aspect='auto', interpolation='nearest')
            plt.colorbar(label="Intensidade (Escala Aplicada)"); plt.title(f"Imagem {dim}x{dim} ({self.last_algorithm_used})")
            filename = f"resultado_{self.username.get()}_{self.algorithm.get()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight'); plt.close(fig)
            return filename
        except Exception as e: self.log(f"Erro ao salvar o gráfico: {e}"); return ""

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