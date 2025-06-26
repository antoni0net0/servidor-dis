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
        self.H_1_CSV = "H-1.csv"
        self.H_2_CSV = "H-2.csv"
        self.H_1_NPY = "H-1.npy"
        self.H_2_NPY = "H-2.npy"
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

        self.H_1 = None
        self.H_2 = None

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
        """Loads H_1 and H_2 matrices from .npy or .csv files."""
        self._clear_screen()
        self._print_title()
        print('Loading base files...')
        h_1_npy_path = os.path.join(self.config.DATA_DIR, self.config.H_1_NPY)
        h_2_npy_path = os.path.join(self.config.DATA_DIR, self.config.H_2_NPY)
        h_1_csv_path = os.path.join(self.config.DATA_DIR, self.config.H_1_CSV)
        h_2_csv_path = os.path.join(self.config.DATA_DIR, self.config.H_2_CSV)

        if os.path.exists(h_1_npy_path) and os.path.exists(h_2_npy_path):
            self.H_1 = np.load(h_1_npy_path)
            self.H_2 = np.load(h_2_npy_path)
            self.logger.info("Matrices loaded from .npy files.")
        else:
            self.logger.info("NPY files not found. Loading from CSV and saving as NPY.")
            self.H_1 = np.genfromtxt(h_1_csv_path, delimiter=',')
            self.H_2 = np.genfromtxt(h_2_csv_path, delimiter=',')
            np.save(h_1_npy_path, self.H_1)
            np.save(h_2_npy_path, self.H_2)
            self.logger.info("Matrices loaded from .csv files and saved as .npy files.")
    
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

        response_parts = []
        if option in [1, 2, 3]:
            response_raw = self._receive_message(client_socket, address)
            if not response_raw:
                return
            response_parts = response_raw.split("-")
            if response_parts[0] != "OK":
                self.logger.warning(f"Client {address} did not send 'OK' for processing option.")
                return

        model = response_parts[2] if len(response_parts) > 2 else ""
        image_model = response_parts[3] if len(response_parts) > 3 else ""
        username = response_parts[1] if len(response_parts) > 1 else ""
        algorithm_model = response_parts[4] if len(response_parts) > 4 else ""

        if option == 1:
            # Option 1: Process with queueing
            if not self.semaphore.acquire(blocking=False):
                self.logger.info(f"Client {address} put in waiting queue.")
                self.waiting_queue.put(th.Thread(target=self._process_client_request, args=(client_socket, address, model, image_model, username, algorithm_model)))
                self._send_message(client_socket, address, 'WAIT-Wait in queue')
                return
            else:
                self.logger.info(f"Client {address} acquired semaphore and will be processed immediately.")
                try:
                    self._send_message(client_socket, address, 'OK-Ready to receive')
                    signal_gain = self._receive_signal_gain(client_socket, address)
                    if signal_gain is not None:
                        self._reconstruct_image(client_socket, address, model, image_model, signal_gain, username, algorithm_model)
                finally:
                    self.semaphore.release()
                    self.logger.info(f"Semaphore released for {address}.")
                    if not self.waiting_queue.empty():
                        next_client_thread = self.waiting_queue.get()
                        next_client_thread.start()
                        self.logger.info(f"Started processing next client from queue.")

        elif option in [2, 3]:
            # Options 2 & 3: Process directly (original behavior without explicit queueing check)
            self._send_message(client_socket, address, 'OK-Ready to receive')
            signal_gain = self._receive_signal_gain(client_socket, address)
            if signal_gain is not None:
                self._reconstruct_image(client_socket, address, model, image_model, signal_gain, username, algorithm_model)
            self._handle_client_options(client_socket, address) # Loop back to options after processing

        elif option == 4:
            # Option 4: Send report
            username = response_parts[1] if len(response_parts) > 1 else ""
            self._send_report(client_socket, address, username)
            self._handle_client_options(client_socket, address) # Loop back to options after sending report

        elif option == 5:
            # Option 5: Disconnect
            if response_parts[0] == "OK":
                self.logger.warning(f"Client disconnected: {address}")
                self._remove_client(client_socket)
                self._send_message(client_socket, address, 'OK-8-Disconnected')
                self._clear_screen()
                self._print_title()
                print(f"{len(self.connected_clients)} client(s) connected...")
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

    def _calculate_cgne(self, g: np.ndarray, model: str) -> tuple[np.ndarray, int]:
        """Calculates image reconstruction using the Conjugate Gradient Normal Equation (CGNE) method."""
        H_matrix = self.H_1 if model == "H_1" else self.H_2
        f = np.zeros(H_matrix.shape[1])
        r = g - np.dot(H_matrix, f)
        p = np.dot(H_matrix.T, r)
        iter_count = 0

        # Progress reporting setup
        total_elements = len(g)
        progress_increment = total_elements // 100
        last_reported_progress = -1

        for i in range(total_elements): # Iterating up to len(g) for progress, but convergence is key
            alpha_numerator = np.dot(r.T, r)
            alpha_denominator = np.dot(p.T, p)
            if alpha_denominator == 0:
                self.logger.warning(f"CGNE: Alpha denominator is zero. Terminating early at iteration {iter_count}.")
                break
            alpha = alpha_numerator / alpha_denominator

            f = f + alpha * p
            r_next = r - alpha * np.dot(H_matrix, p)

            error = abs(np.linalg.norm(r, ord=2) - np.linalg.norm(r_next, ord=2))
            if error < 1e-4:
                self.logger.info(f"CGNE: Error less than 1e-4 at iteration {iter_count}.")
                break

            if iter_count >= 10 and error > 1e-3: # Allow more iterations if not converged quickly
                self.logger.info(f"CGNE: Passed 10 iterations with error {error}. Continuing.")

            beta_numerator = np.dot(r_next.T, r_next)
            beta_denominator = np.dot(r.T, r)
            if beta_denominator == 0:
                self.logger.warning(f"CGNE: Beta denominator is zero. Terminating early at iteration {iter_count}.")
                break
            beta = beta_numerator / beta_denominator

            p = beta * p + np.dot(H_matrix.T, r_next)
            r = r_next
            iter_count += 1

            current_progress = i // progress_increment if progress_increment > 0 else 0
            if current_progress > last_reported_progress:
                last_reported_progress = current_progress
                self._clear_screen()
                self._print_title()
                print(f'Processing: {last_reported_progress}% of {total_elements} packages')
                self.logger.info(f'Processing: {last_reported_progress}% of {total_elements} packages')

        print('Processing finished')
        self.logger.info("CGNE processing finished.")
        return f, iter_count

    def _calculate_cgnr(self, g: np.ndarray, model: str) -> tuple[np.ndarray, int]:
        """Calculates image reconstruction using the Conjugate Gradient Normal Residual (CGNR) method."""
        H_matrix = self.H_1 if model == "H_1" else self.H_2
        f = np.zeros(H_matrix.shape[1])
        r = g - np.dot(H_matrix, f)
        z = np.dot(H_matrix.T, r)
        p = z
        iter_count = 0

        # Progress reporting setup
        total_elements = len(g)
        progress_increment = total_elements // 100
        last_reported_progress = -1

        for i in range(total_elements): # Iterating up to len(g) for progress, but convergence is key
            w = np.dot(H_matrix, p)
            alpha_numerator = np.dot(z.T, z)
            alpha_denominator = np.dot(w.T, w)
            if alpha_denominator == 0:
                self.logger.warning(f"CGNR: Alpha denominator is zero. Terminating early at iteration {iter_count}.")
                break
            alpha = alpha_numerator / alpha_denominator

            f = f + alpha * p
            r_next = r - alpha * w
            z_next = np.dot(H_matrix.T, r_next)

            error = abs(np.linalg.norm(r, ord=2) - np.linalg.norm(r_next, ord=2))
            if error < 1e-4:
                self.logger.info(f"CGNR: Error less than 1e-4 at iteration {iter_count}.")
                break

            beta_numerator = np.dot(z_next.T, z_next)
            beta_denominator = np.dot(z.T, z)
            if beta_denominator == 0:
                self.logger.warning(f"CGNR: Beta denominator is zero. Terminating early at iteration {iter_count}.")
                break
            beta = beta_numerator / beta_denominator

            p = z_next + beta * p
            r = r_next
            z = z_next
            iter_count += 1

            current_progress = i // progress_increment if progress_increment > 0 else 0
            if current_progress > last_reported_progress:
                last_reported_progress = current_progress
                self._clear_screen()
                self._print_title()
                print(f'Processing: {last_reported_progress}% of {total_elements} packages')
                self.logger.info(f'Processing: {last_reported_progress}% of {total_elements} packages')

        print('Processing finished')
        self.logger.info("CGNR processing finished.")
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
        confirmation to the client.
        """
        process = psutil.Process(os.getpid())

        start_datetime = datetime.datetime.now()
        start_time = time.time()
        start_cpu_percent = process.cpu_percent(interval=None) # Non-blocking
        start_memory_rss = process.memory_info().rss

        reconstructed_image_data = None
        iterations = 0

        if algorithm_model == "CGNE":
            reconstructed_image_data, iterations = self._calculate_cgne(signal_gain, model)
        elif algorithm_model == "CGNR":
            reconstructed_image_data, iterations = self._calculate_cgnr(signal_gain, model)
        else:
            self.logger.error(f"Unknown algorithm model: {algorithm_model}")
            self._send_message(client_socket, address, "ERROR-Unknown algorithm")
            return

        if reconstructed_image_data is None:
            self.logger.error(f"Image reconstruction failed for {address}.")
            self._send_message(client_socket, address, "ERROR-Image reconstruction failed")
            return

        # Reshape the 1D array back into a square image
        image_dim = int(np.sqrt(len(reconstructed_image_data)))
        if image_dim * image_dim != len(reconstructed_image_data):
            self.logger.error(f"Reconstructed data length {len(reconstructed_image_data)} is not a perfect square for image reshaping.")
            # Handle error, perhaps pad or resize if necessary, or just send error to client
            self._send_message(client_socket, address, "ERROR-Invalid image data size")
            return
            
        reconstructed_image = reconstructed_image_data.reshape((image_dim, image_dim), order='F') # 'F' for Fortran-like order (column-major)

        end_datetime = datetime.datetime.now()
        end_time = time.time()
        end_cpu_percent = process.cpu_percent(interval=None) # Non-blocking
        end_memory_rss = process.memory_info().rss

        total_processing_time = end_time - start_time
        # CPU usage might need a small interval to be accurate if measured this way.
        # psutil.cpu_percent() measures system-wide CPU usage over an interval if interval is not None.
        # For per-process usage, you might want to consider `process.cpu_times()` for start/end.
        # For simplicity, keeping the original logic, but note this might not be precise per process.
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

        plt.figure(figsize=(8, 6)) # Create a new figure
        plt.imshow(reconstructed_image, cmap='gray')
        plt.title('Performance Report and Reconstructed Image')
        # Adjust text position based on figure coordinates (0 to 1)
        plt.gcf().text(0.02, 0.5, report_info, fontsize=9, color='white', ha='left', va='center',
                       bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.5'))
        plt.axis('off') # Hide axes for a cleaner look
        plt.tight_layout(rect=[0.25, 0, 1, 1]) # Adjust layout to make space for text
        plt.savefig(report_filepath)
        plt.close() # Close the figure to free up memory

        self.logger.info(f"Performance report saved to {report_filepath}")
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