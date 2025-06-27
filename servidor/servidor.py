import matplotlib.pyplot as plt
import threading as th
import logging
import hashlib
import numpy as np
import socket
import time
import datetime
import psutil
import csv
import os
import queue


# TODO: ajeitar o menu na hora de escolher o csv, ele ta errado, botar somente os que tem uma coluna (todos menos os de H_1 e H_2)

class ServerConfig:
    """
    Configuration class for server parameters.
    Encapsulates all fixed parameters for easy modification and readability.
    """
    def __init__(self):
        self.SERVER_NAME = ''
        self.SERVER_PORT = 6000
        self.BUFFER_SIZE = 2048
        self.MAX_SIMULTANEOUS_CLIENTS = 3
        
        self.LOG_DIR = "./log" 
        self.LOG_FILE = os.path.join(self.LOG_DIR, "servidor.log")
        
        self.CONTENT_DIR = "./content"
        self.DATA_DIR = "./data"
        # Lista de arquivos de matriz disponíveis (nome base, sem extensão)
        self.MATRIX_FILES = [
            "H_1", "H_2", "G_1", "G_2", "g_30x30_1", "g_30x30_2", "A_30x30_1", "A_60x60_1"
        ]
class Server:
    """
    Implements a multi-threaded server for image reconstruction.

    This server handles client connections, processes image reconstruction
    requests using CGNE or CGNR algorithms, and provides performance reports.
    It manages concurrent clients using a semaphore and a queue.
    """
    def __init__(self):
        self.config = ServerConfig() # Initialize self.config FIRST

        self._create_required_directories() # Now self.config exists when this is called
        self._setup_logging() # Now self.logger will be initialized correctly

        self.server_address = (self.config.SERVER_NAME, self.config.SERVER_PORT)
        self.connected_clients = [] # Initialize here to prevent AttributeError in __del__ if init fails early

        # Dicionário para armazenar todas as matrizes carregadas
        self.matrices = {}

        self.semaphore = th.Semaphore(self.config.MAX_SIMULTANEOUS_CLIENTS)
        self.waiting_queue = queue.Queue()

        self.server_socket = self._initialize_server_socket()
        self._load_matrices()

    def _create_required_directories(self):
        """Ensures that necessary directories (log, content, data) exist."""
        os.makedirs(self.config.LOG_DIR, exist_ok=True)
        os.makedirs(self.config.CONTENT_DIR, exist_ok=True)
        os.makedirs(self.config.DATA_DIR, exist_ok=True)
        # We can't log this with self.logger yet, as logging is not set up
        print(f"Ensured directories exist: {self.config.LOG_DIR}, {self.config.CONTENT_DIR}, {self.config.DATA_DIR}")


    def _setup_logging(self):
        """Configures the logging for the server."""
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            filename=self.config.LOG_FILE,
            encoding="utf-8",
            level=logging.INFO,
            format="%(levelname)s - %(asctime)s: %(message)s"
        )
        self.logger.info("Logger initialized.")

    def _initialize_server_socket(self):
        """Initializes and binds the server socket."""
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.bind(self.server_address)
            server_socket.listen()
            self.logger.info(f"Server socket created on port: '{self.config.SERVER_PORT}'")
            return server_socket
        except Exception as e:
            self.logger.critical(f"Failed to initialize server socket: {e}")
            raise

    def _load_matrices(self):
        """Carrega todas as matrizes disponíveis (npy ou csv) para reconstrução de imagem."""
        self._clear_screen()
        self._print_title()
        print('Carregando arquivos de matriz disponíveis...')
        for base_name in self.config.MATRIX_FILES:
            npy_path = os.path.join(self.config.DATA_DIR, f"{base_name}.npy")
            csv_path = os.path.join(self.config.DATA_DIR, f"{base_name}.csv")
            try:
                if os.path.exists(npy_path):
                    try:
                        self.matrices[base_name] = np.load(npy_path)
                        self.logger.info(f"Matriz '{base_name}' carregada de arquivo .npy.")
                        print(f"[OK] {base_name}.npy carregado com shape {self.matrices[base_name].shape}")
                    except Exception as e:
                        self.logger.error(f"Erro ao carregar '{base_name}.npy': {e}")
                        print(f"[ERRO] Falha ao carregar {base_name}.npy: {e}")
                elif os.path.exists(csv_path):
                    try:
                        self.matrices[base_name] = np.genfromtxt(csv_path, delimiter=',')
                        np.save(npy_path, self.matrices[base_name])
                        self.logger.info(f"Matriz '{base_name}' carregada de .csv e salva como .npy.")
                        print(f"[OK] {base_name}.csv carregado com shape {self.matrices[base_name].shape}")
                    except Exception as e:
                        self.logger.error(f"Erro ao carregar '{base_name}.csv': {e}")
                        print(f"[ERRO] Falha ao carregar {base_name}.csv: {e}")
                else:
                    self.logger.warning(f"Arquivo de matriz '{base_name}' não encontrado em .npy nem .csv.")
                    print(f"[AVISO] {base_name} não encontrado em .npy nem .csv")
            except Exception as e:
                self.logger.error(f"Erro inesperado ao tentar carregar '{base_name}': {e}")
                print(f"[ERRO] Erro inesperado ao tentar carregar {base_name}: {e}")
        print(f"Matrizes disponíveis: {list(self.matrices.keys())}")
    
    def __del__(self):
        """Cleans up resources when the server object is deleted."""
        # Check if logger exists before using it
        if hasattr(self, 'logger'):
            self.logger.info("Server socket finalized!")
        
        # Ensure connected_clients exists before iterating
        if hasattr(self, 'connected_clients'):
            for client_socket, _ in self.connected_clients:
                try:
                    client_socket.close()
                except Exception as e:
                    if hasattr(self, 'logger'):
                        self.logger.error(f"Error closing client socket: {e}")
            self.connected_clients.clear()
        
        if hasattr(self, 'server_socket') and self.server_socket:
            try:
                self.server_socket.close()
            except Exception as e:
                if hasattr(self, 'logger'):
                    self.logger.error(f"Error closing server socket: {e}")
        self.H_1 = None
        self.H_2 = None
        
        if hasattr(self, 'logger'):
            self.logger.info("Server resources released.")


    def _clear_screen(self):
        """Clears the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def _print_title(self) -> None:
        """Prints the server title to the console."""
        print("--------------------")
        print("      SERVIDOR")
        print("--------------------\n")

    def _send_message(self, client_socket: socket.socket, address: tuple, message: str) -> None:
        """Sends a message to a client."""
        try:
            client_socket.send(message.encode())
            self.logger.info(f"Recipient: {address} - Sent: '{message}'")
        except Exception as e:
            self.logger.error(f"Failed to send message to {address}. Client removed: {e}")
            self._remove_client(client_socket)

    def _receive_message(self, client_socket: socket.socket, address: tuple) -> str:
        """Receives a message from a client."""
        try:
            message = client_socket.recv(self.config.BUFFER_SIZE).decode('utf-8')
            self.logger.info(f"Sender: {address} - Received: '{message}'")
            return message
        except Exception as e:
            self.logger.error(f"Failed to receive message from {address}. Client removed: {e}")
            self._remove_client(client_socket)
            return "" # Return empty string or raise an exception to indicate failure

    def _remove_client(self, client_socket: socket.socket) -> None:
        """Removes a client from the connected clients list."""
        # Find and remove the client by its socket object
        self.connected_clients = [c for c in self.connected_clients if c[0] != client_socket]
        try:
            client_socket.close()
        except Exception as e:
            self.logger.error(f"Error closing socket during client removal: {e}")


    def _start_server_prompt(self) -> bool:
        """Prompts the user to start the server."""
        while True:
            self._clear_screen()
            self._print_title()
            choice = input("Do you want to start the server [S/N] ? ").lower().strip()
            if choice in ['s', 'sim']:
                self.logger.info("Server was initialized!")
                return True
            elif choice in ['n', 'não']:
                self.logger.info("Server was not initialized!")
                return False
            else:
                print('Invalid choice. Please select S or N.')
                self.logger.warning("Invalid server initialization response.")
                time.sleep(2)

    def _process_client_request(self, client_socket: socket.socket, address: tuple, model: str, image_model: str, username: str, algorithm_model: str) -> None:
        """
        Processes a client's image reconstruction request.
        This function is executed after acquiring the semaphore.
        """
        try:
            self._send_message(client_socket, address, 'OK-Ready to receive')
            signal_gain = self._receive_signal_gain(client_socket, address)
            if signal_gain is not None:
                self._reconstruct_image(client_socket, address, model, image_model, signal_gain, username, algorithm_model)
        finally:
            self.semaphore.release()
            self.logger.info(f"Semaphore released for {address}.")
            # If there are clients waiting, start the next one
            if not self.waiting_queue.empty():
                next_client_thread = self.waiting_queue.get()
                next_client_thread.start()
                self.logger.info(f"Started processing next client from queue.")

    def _handle_client_options(self, client_socket: socket.socket, address: tuple) -> None:
        """Handles the different options a connected client can choose."""
        self._clear_screen()
        self._print_title()
        print(f"{len(self.connected_clients)} client(s) connected...")

        client_option_raw = self._receive_message(client_socket, address)
        if not client_option_raw: # Handle case where receive fails
            return

        client_option_parts = client_option_raw.split("-")
        option = 0
        if client_option_parts[0] == 'OPTION':
            try:
                option = int(client_option_parts[1])
            except ValueError:
                self.logger.error(f"Invalid option received from {address}: {client_option_parts[1]}")
                self._send_message(client_socket, address, "ERROR-Invalid option")
                return


        # NOVO FLUXO: escolha de modelagem antes do algoritmo
        if option in [1, 2, 3]:
            # 1. Envie lista de matrizes disponíveis (2D)
            available_matrices = [k for k, v in self.matrices.items() if v is not None and len(v.shape) == 2]
            self._send_message(client_socket, address, "MATRICES-" + ",".join(available_matrices))
            chosen_matrix = self._receive_message(client_socket, address)
            if chosen_matrix not in available_matrices:
                self._send_message(client_socket, address, "ERROR-Matriz inválida")
                return

            # 2. Envie lista de vetores disponíveis (arquivos 1D)
            data_files = os.listdir(self.config.DATA_DIR)
            available_vectors = [f[:-4] for f in data_files if (f.endswith('.npy') or f.endswith('.csv')) and f[:-4] not in available_matrices]
            self._send_message(client_socket, address, "VECTORS-" + ",".join(available_vectors))
            chosen_vector = self._receive_message(client_socket, address)
            if chosen_vector not in available_vectors:
                self._send_message(client_socket, address, "ERROR-Vetor inválido")
                return

            # 3. Recebe algoritmo
            self._send_message(client_socket, address, "ALGORITHMS-CGNE,CGNR")
            chosen_algorithm = self._receive_message(client_socket, address)
            if chosen_algorithm not in ["CGNE", "CGNR"]:
                self._send_message(client_socket, address, "ERROR-Algoritmo inválido")
                return

            # 4. Recebe username e imagem
            self._send_message(client_socket, address, "SEND-USER-IMAGE")
            user_image = self._receive_message(client_socket, address)
            user_image_parts = user_image.split('-')
            username = user_image_parts[0] if len(user_image_parts) > 0 else ""
            image_model = user_image_parts[1] if len(user_image_parts) > 1 else ""

            # Carrega matriz e vetor
            matriz = self.matrices.get(chosen_matrix)
            vetor_path_npy = os.path.join(self.config.DATA_DIR, f"{chosen_vector}.npy")
            vetor_path_csv = os.path.join(self.config.DATA_DIR, f"{chosen_vector}.csv")
            if os.path.exists(vetor_path_npy):
                vetor = np.load(vetor_path_npy)
            elif os.path.exists(vetor_path_csv):
                vetor = np.loadtxt(vetor_path_csv, delimiter=',')
            else:
                self._send_message(client_socket, address, "ERROR-Vetor não encontrado")
                return

            # Processa
            if chosen_algorithm == "CGNE":
                resultado, iterations = self._calculate_cgne(vetor, matriz)
            else:
                resultado, iterations = self._calculate_cgnr(vetor, matriz)

            # Aqui você pode seguir com o fluxo normal de salvar imagem, enviar resposta, etc.
            self._send_message(client_socket, address, f"OK-Process finished-{iterations}")
            return

        elif option == 4:
            # Option 4: Send report
            username = response_parts[1] if len(response_parts) > 1 else ""
            self._send_report(client_socket, address, username)
            self._handle_client_options(client_socket, address) # Loop back to options after sending report

        elif option == 5:
            # Option 5: Disconnect
            if response_parts[0] == "OK":
                self._send_message(client_socket, address, 'OK-8-Desconectado')
                self._remove_client(client_socket)
                return
        else:
            self.logger.warning(f"Unknown option {option} from client {address}.")
            self._send_message(client_socket, address, "ERROR-Unknown option")


    def _calculate_file_checksum(self, filename: str) -> str:
        """Calculates the MD5 checksum of a file."""
        checksum = hashlib.md5()
        file_path = os.path.join(self.config.CONTENT_DIR, filename)
        if not os.path.exists(file_path):
            self.logger.error(f"File not found for checksum: {file_path}")
            return ""

        with open(file_path, "rb") as file:
            while data := file.read(self.config.BUFFER_SIZE):
                checksum.update(data)
        return checksum.hexdigest()

    def _send_report(self, client_socket: socket.socket, address: tuple, username: str) -> None:
        """Sends a performance report file to the client."""
        filename_to_send = self._get_user_report_filename(client_socket, address, username)
        if not filename_to_send:
            self.logger.warning(f"No report file to send for user {username} to {address}.")
            return

        file_path = os.path.join(self.config.CONTENT_DIR, filename_to_send)
        if not os.path.exists(file_path):
            self.logger.error(f"Report file {file_path} does not exist.")
            self._send_message(client_socket, address, "ERROR-File not found on server")
            return

        file_size = os.path.getsize(file_path)
        num_packets = (file_size // self.config.BUFFER_SIZE) + (1 if file_size % self.config.BUFFER_SIZE else 0)
        num_digits = len(str(num_packets))
        # This calculation seems a bit off, it might be an attempt to calculate max packet size
        # num_buffer = num_digits + 1 + 16 + 1 + self.config.BUFFER_SIZE # original comment
        checksum = self._calculate_file_checksum(filename_to_send)

        self._send_message(client_socket, address, f"OK-2-{num_packets}-{num_digits}-{checksum}")
        
        start_confirmation = self._receive_message(client_socket, address).split("-")
        if start_confirmation[0] != "OK":
            self.logger.warning(f"Client {address} did not confirm report transfer start.")
            return

        with open(file_path, "rb") as file:
            for i in range(num_packets):
                data = file.read(self.config.BUFFER_SIZE)
                if not data:
                    break

                data_checksum = hashlib.md5(data).digest()
                # Format: packet_index-checksum-data
                packet_data = b" ".join([f"{i:{'0'}{num_digits}}".encode(), data_checksum, data])

                try:
                    client_socket.send(packet_data)
                    self.logger.info(f"Recipient: {address} - Sent: 'Packet {i+1}'")
                except Exception as e:
                    self.logger.error(f"Error sending packet {i+1} to {address}. Client removed: {e}")
                    self._remove_client(client_socket)
                    break

                # Wait for ACK/NACK for the sent packet
                ack_status = self._receive_message(client_socket, address)
                while ack_status == "NOK":
                    try:
                        client_socket.send(packet_data)
                        self.logger.warning(f"Recipient: {address} - Resent: 'Packet {i+1}'")
                        ack_status = self._receive_message(client_socket, address)
                    except Exception as e:
                        self.logger.error(f"Error re-sending packet {i+1} to {address}. Client removed: {e}")
                        self._remove_client(client_socket)
                        break
            else:
                self.logger.info(f"'OK-4-All {num_packets} packets sent!' to {address}")
                return

    def _get_user_report_filename(self, client_socket: socket.socket, address: tuple, username: str) -> str:
        """Guides the client through selecting a report file."""
        self._clear_screen()
        self._print_title()

        available_files = [f for f in os.listdir(self.config.CONTENT_DIR) if username in f]
        num_available_files = len(available_files)

        self._send_message(client_socket, address, str(num_available_files))
        
        confirmation_size = self._receive_message(client_socket, address).split("-")
        
        if confirmation_size[0] == "ERROR":
            self.logger.error(f"Client {address} reported an error in request (ERR-1).")
            print("Request Error")
            time.sleep(2)
            return ""
        
        if num_available_files <= 0:
            self.logger.error(f"No files available for user {username} on server (ERR-2).")
            print("No files on server for this user.")
            time.sleep(2)
            return ""
            
        for i, filename in enumerate(available_files):
            self._send_message(client_socket, address, filename)
            ack = self._receive_message(client_socket, address).split("-")
            if len(ack) < 2 or ack[1] != str(i + 1):
                self.logger.warning(f"Client {address} did not acknowledge file list correctly for {filename}.")
                # Depending on robustness needed, you might want to break or retry here
                break
        
        while True:
            chosen_filename = self._receive_message(client_socket, address)
            if chosen_filename not in available_files:
                self._send_message(client_socket, address, "ERROR-3-File not found!")
                self.logger.warning(f"Client {address} requested non-existent file: {chosen_filename}")
            else:
                self._send_message(client_socket, address, 'OK-1-Confirmation')
                self.logger.info(f"Client {address} selected file: {chosen_filename}")
                return chosen_filename

    def _calculate_cgne(self, g: np.ndarray, H_matrix: np.ndarray) -> tuple[np.ndarray, int]:
        """Calcula reconstrução de imagem usando CGNE, recebendo H e g explicitamente."""
        if H_matrix is None:
            raise ValueError("Matriz H não carregada.")
        if g is None:
            raise ValueError("Vetor g não fornecido.")
        if len(H_matrix.shape) != 2:
            raise ValueError(f"Matriz H deve ser 2D, shape atual: {H_matrix.shape}")
        if g.shape[0] != H_matrix.shape[0]:
            raise ValueError(f"Vetor g deve ter o mesmo número de linhas que H. g: {g.shape}, H: {H_matrix.shape}")
        f = np.zeros(H_matrix.shape[1])
        r = g - H_matrix @ f
        p = H_matrix.T @ r
        iter_count = 0
        max_iter = 100
        for i in range(max_iter):
            alpha = np.dot(r, r) / np.dot(p, p)
            f = f + alpha * p
            r_next = r - alpha * (H_matrix @ p)
            if np.abs(np.linalg.norm(r) - np.linalg.norm(r_next)) < 1e-4:
                break
            beta = np.dot(r_next, r_next) / np.dot(r, r)
            p = beta * p + H_matrix.T @ r_next
            r = r_next
            iter_count += 1
        return f, iter_count

    def _calculate_cgnr(self, g: np.ndarray, model: str) -> tuple[np.ndarray, int]:
        """Calcula reconstrução de imagem usando CGNR."""
        H_matrix = self.matrices.get(model)
        if H_matrix is None:
            raise ValueError(f"Matriz '{model}' não carregada.")
        f = np.zeros(H_matrix.shape[1])
        r = g - np.dot(H_matrix, f)
        z = np.dot(H_matrix.T, r)
        p = z
        iter_count = 0

        max_iter = 100  # Limite de iterações para evitar lentidão

        for i in range(max_iter):
            w = np.dot(H_matrix, p)
            alpha_numerator = np.dot(z.T, z)
            alpha_denominator = np.dot(w.T, w)
            if alpha_denominator == 0:
                break
            alpha = alpha_numerator / alpha_denominator

            f = f + alpha * p
            r_next = r - alpha * w
            z_next = np.dot(H_matrix.T, r_next)

            error = abs(np.linalg.norm(r, ord=2) - np.linalg.norm(r_next, ord=2))
            if error < 1e-4:
                break

            beta_numerator = np.dot(z_next.T, z_next)
            beta_denominator = np.dot(z.T, z)
            if beta_denominator == 0:
                break
            beta = beta_numerator / beta_denominator

            p = z_next + beta * p
            r = r_next
            z = z_next
            iter_count += 1

        return f, iter_count

    def _receive_signal_gain(self, client_socket: socket.socket, address: tuple) -> np.ndarray | None:
        """Receives signal gain data (numpy array) from the client."""
        try:
            data_size_bytes = client_socket.recv(8)
            if not data_size_bytes:
                self.logger.error(f"Client {address} disconnected during signal gain size reception.")
                return None
            size = int.from_bytes(data_size_bytes, byteorder='big')
            self.logger.info(f"Sender: {address} - Received: 'Data size {size}'")

            received_data = bytearray()
            packets_received = 0
            while len(received_data) < size:
                chunk = client_socket.recv(4096) # Receive in chunks
                if not chunk:
                    self.logger.error(f"Client {address} disconnected during signal gain data reception.")
                    break
                received_data.extend(chunk)
                packets_received += 1
                # Acknowledge each chunk if necessary for flow control, though not in original
                # self._send_message(client_socket, address, f"ACK-{packets_received}")

            if len(received_data) == size:
                self.logger.info(f"Successfully received all {packets_received} packets from {address}.")
                return np.frombuffer(received_data, dtype=np.float64)
            else:
                self.logger.error(f"Incomplete data received from {address}. Expected {size}, got {len(received_data)}.")
                return None
        except Exception as e:
            self.logger.error(f"Error receiving signal gain from {address}: {e}")
            self._remove_client(client_socket)
            return None

    def _reconstruct_image(self, client_socket: socket.socket, address: tuple, model: str, image_model: str, signal_gain: np.ndarray, username: str, algorithm_model: str) -> None:
        """
        Reconstructs the image, saves a performance report, and sends a
        confirmation to the client. Agora com logs detalhados de shapes e erros.
        """
        process = psutil.Process(os.getpid())

        start_datetime = datetime.datetime.now()
        start_time = time.time()
        start_cpu_percent = process.cpu_percent(interval=None) # Non-blocking
        start_memory_rss = process.memory_info().rss

        self.logger.info(f"[RECON] Usuário: {username}, Modelo: {model}, Algoritmo: {algorithm_model}, Imagem: {image_model}")
        # Log shapes das matrizes e do sinal recebido
        matriz = self.matrices.get(model)
        if matriz is None:
            self.logger.error(f"[RECON] Matriz '{model}' não carregada!")
            self._send_message(client_socket, address, f"ERROR-Matriz '{model}' não carregada")
            return
        self.logger.info(f"[RECON] Shape da matriz '{model}': {matriz.shape}")
        self.logger.info(f"[RECON] Shape do signal_gain recebido: {signal_gain.shape}")

        if signal_gain.shape[0] != matriz.shape[0]:
            self.logger.error(f"[RECON] Tamanho do signal_gain ({signal_gain.shape[0]}) diferente do número de linhas da matriz ({matriz.shape[0]})!")
            self._send_message(client_socket, address, f"ERROR-Tamanho do signal_gain ({signal_gain.shape[0]}) diferente do esperado ({matriz.shape[0]})")
            return

        reconstructed_image_data = None
        iterations = 0

        try:
            if algorithm_model == "CGNE":
                reconstructed_image_data, iterations = self._calculate_cgne(signal_gain, model)
            elif algorithm_model == "CGNR":
                reconstructed_image_data, iterations = self._calculate_cgnr(signal_gain, model)
            else:
                self.logger.error(f"Unknown algorithm model: {algorithm_model}")
                self._send_message(client_socket, address, "ERROR-Unknown algorithm")
                return
        except Exception as e:
            self.logger.error(f"[RECON] Erro ao executar algoritmo {algorithm_model}: {e}")
            self._send_message(client_socket, address, f"ERROR-Falha ao executar algoritmo: {e}")
            return

        if reconstructed_image_data is None:
            self.logger.error(f"Image reconstruction failed for {address}.")
            self._send_message(client_socket, address, "ERROR-Image reconstruction failed")
            return

        # Reshape the 1D array back into a square image
        image_dim = int(np.sqrt(len(reconstructed_image_data)))
        if image_dim * image_dim != len(reconstructed_image_data):
            self.logger.error(f"[RECON] Reconstructed data length {len(reconstructed_image_data)} is not a perfect square for image reshaping.")
            self._send_message(client_socket, address, f"ERROR-Invalid image data size: {len(reconstructed_image_data)}")
            return

        try:
            reconstructed_image = reconstructed_image_data.reshape((image_dim, image_dim), order='F') # 'F' for Fortran-like order (column-major)
        except Exception as e:
            self.logger.error(f"[RECON] Erro ao fazer reshape da imagem: {e}")
            self._send_message(client_socket, address, f"ERROR-Reshape: {e}")
            return

        end_datetime = datetime.datetime.now()
        end_time = time.time()
        end_cpu_percent = process.cpu_percent(interval=None) # Non-blocking
        end_memory_rss = process.memory_info().rss

        total_processing_time = end_time - start_time
        cpu_usage_diff = end_cpu_percent - start_cpu_percent
        memory_usage_mb = (end_memory_rss - start_memory_rss) / (1024 ** 2)

        report_info = (
            f"User: {username}\n"
            f"Iterations: {iterations}\n"
            f"Start Time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"End Time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Total Time (s): {total_processing_time:.2f}\n"
            f"CPU Usage (%): {cpu_usage_diff:.2f}\n"
            f"Memory Usage (MB): {memory_usage_mb:.2f}"
        )

        report_filename = f"{username}-{model}-{image_model}-{algorithm_model}.png"
        report_filepath = os.path.join(self.config.CONTENT_DIR, report_filename)

        try:
            plt.figure(figsize=(8, 6)) # Create a new figure
            plt.imshow(reconstructed_image, cmap='gray')
            plt.title('Performance Report and Reconstructed Image')
            plt.gcf().text(0.02, 0.5, report_info, fontsize=9, color='white', ha='left', va='center',
                        bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.5'))
            plt.axis('off') # Hide axes for a cleaner look
            plt.tight_layout(rect=[0.25, 0, 1, 1]) # Adjust layout to make space for text
            plt.savefig(report_filepath)
            plt.close() # Close the figure to free up memory
            self.logger.info(f"Performance report saved to {report_filepath}")
        except Exception as e:
            self.logger.error(f"[RECON] Erro ao salvar imagem: {e}")
            self._send_message(client_socket, address, f"ERROR-Falha ao salvar imagem: {e}")
            return

        self._send_message(client_socket, address, 'OK-Process finished')

    def run(self) -> None:
        """Main method to run the server."""
        self._clear_screen()
        self._print_title()

        if not self._start_server_prompt():
            self.logger.info("Server not started by user.")
            return

        self._clear_screen()
        self._print_title()
        print('Waiting for client connections...')

        while True:
            try:
                client_socket, address = self.server_socket.accept()
                self.connected_clients.append((client_socket, address))
                self.logger.info(f"New client connected: {address}. Total clients: {len(self.connected_clients)}")
                # Start a new thread to handle the client's requests
                client_handler_thread = th.Thread(
                    target=self._handle_client_options,
                    args=(client_socket, address),
                    daemon=True # Daemon threads exit when the main program exits
                )
                client_handler_thread.start()
            except KeyboardInterrupt:
                self.logger.info("Server shutting down due to user interrupt (Ctrl+C).")
                break
            except Exception as e:
                self.logger.error(f"Error accepting new connection: {e}")
                # Optionally, add a small sleep to prevent busy-looping on errors
                time.sleep(1)

if __name__ == "__main__":
    server = Server()
    server.run()