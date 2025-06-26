import socket
import threading
import os
import time
import numpy as np
import hashlib
import json # For potential future structured messages
import matplotlib.pyplot as plt
import logging

class ClientConfig:
    """
    Configuration class for client parameters.
    """
    def __init__(self):
        self.SERVER_HOST = '127.0.0.1' # Loopback address for local testing
        self.SERVER_PORT = 6000
        self.BUFFER_SIZE = 2048
        self.DOWNLOAD_DIR = "./downloads"
        self.LOG_FILE = "./log/cliente.log"

class Client:
    """
    Implements a client application to interact with the Server.

    This client can send image reconstruction requests (with different models
    and algorithms) and retrieve performance reports.
    """
    def __init__(self):
        self.config = ClientConfig()
        self._setup_logging()

        self.client_socket = None
        self.username = "" # To be set by the user
        self.is_connected = False

        self._create_download_directory()

    def _setup_logging(self):
        """Configures the logging for the client."""
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            filename=self.config.LOG_FILE,
            encoding="utf-8",
            level=logging.INFO,
            format="%(levelname)s - %(asctime)s: %(message)s"
        )
        self.logger.info("Client logger initialized.")

    def _create_download_directory(self):
        """Creates the download directory if it doesn't exist."""
        os.makedirs(self.config.DOWNLOAD_DIR, exist_ok=True)
        self.logger.info(f"Download directory '{self.config.DOWNLOAD_DIR}' ensured.")

    def _clear_screen(self):
        """Clears the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def _print_title(self) -> None:
        """Prints the client title to the console."""
        print("--------------------")
        print("       CLIENTE")
        print("--------------------\n")

    def _send_message(self, message: str) -> None:
        """Sends a message to the server."""
        try:
            self.client_socket.send(message.encode('utf-8'))
            self.logger.info(f"Sent: '{message}'")
        except socket.error as e:
            self.logger.error(f"Error sending message: {e}")
            self.is_connected = False
            self._handle_disconnect()

    def _receive_message(self) -> str:
        """Receives a message from the server."""
        try:
            message = self.client_socket.recv(self.config.BUFFER_SIZE).decode('utf-8')
            self.logger.info(f"Received: '{message}'")
            return message
        except socket.error as e:
            self.logger.error(f"Error receiving message: {e}")
            self.is_connected = False
            self._handle_disconnect()
            return ""

    def _handle_disconnect(self):
        """Handles client disconnection."""
        if self.client_socket:
            self.client_socket.close()
            self.client_socket = None
        self.is_connected = False
        print("\nDisconnected from server. Please press Enter to return to main menu.")
        self.logger.info("Client disconnected from server.")

    def connect_to_server(self) -> bool:
        """Attempts to connect to the server."""
        self._clear_screen()
        self._print_title()
        if self.is_connected:
            print("Already connected to the server.")
            time.sleep(1)
            return True

        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            print(f"Attempting to connect to {self.config.SERVER_HOST}:{self.config.SERVER_PORT}...")
            self.client_socket.connect((self.config.SERVER_HOST, self.config.SERVER_PORT))
            self.is_connected = True
            self.logger.info(f"Successfully connected to {self.config.SERVER_HOST}:{self.config.SERVER_PORT}")
            print("Connection successful!")
            time.sleep(1)
            self._clear_screen()
            self._print_title()
            self.username = input("Enter your username: ").strip()
            if not self.username:
                self.username = f"guest_{int(time.time())}"
                print(f"No username entered, using default: {self.username}")
            self.logger.info(f"Username set to: {self.username}")
            return True
        except socket.error as e:
            self.logger.error(f"Failed to connect to server: {e}")
            print(f"Failed to connect to server: {e}")
            self.is_connected = False
            time.sleep(2)
            return False

    def display_options(self) -> str:
        """Displays client options and gets user choice."""
        self._clear_screen()
        self._print_title()
        print(f"Welcome, {self.username}!")
        print("Choose an option:")
        print("1. Reconstruct Image (Queueing - limited simultaneous clients)")
        print("2. Reconstruct Image (No Queueing - direct processing)")
        print("3. Reconstruct Image (Another direct processing - similar to 2)") # Kept for consistency with server options
        print("4. Get Performance Report")
        print("5. Disconnect")
        choice = ""
        while choice not in ['1', '2', '3', '4', '5']:
            choice = input("Enter your choice: ").strip()
            if choice not in ['1', '2', '3', '4', '5']:
                print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")
        return choice

    def _choose_reconstruction_parameters(self) -> tuple[str, str, str]:
        """Guides the user through choosing image reconstruction parameters."""
        self._clear_screen()
        self._print_title()
        
        print("Choose a Model (H_1 or H_2):")
        model = ""
        while model.upper() not in ["H_1", "H_2"]:
            model = input("Enter H_1 or H_2: ").strip()
            if model.upper() not in ["H_1", "H_2"]:
                print("Invalid model. Please enter H_1 or H_2.")
        
        print("\nChoose an Image Model (e.g., 'phantom', 'shepp_logan'):")
        image_model = input("Enter image model name (e.g., 'phantom'): ").strip()
        if not image_model:
            image_model = "default_image" # Fallback
            print(f"No image model entered, using default: {image_model}")

        print("\nChoose an Algorithm Model (CGNE or CGNR):")
        algorithm_model = ""
        while algorithm_model.upper() not in ["CGNE", "CGNR"]:
            algorithm_model = input("Enter CGNE or CGNR: ").strip()
            if algorithm_model.upper() not in ["CGNE", "CGNR"]:
                print("Invalid algorithm. Please enter CGNE or CGNR.")
        
        return model.upper(), image_model, algorithm_model.upper()


    def _generate_dummy_signal_gain(self, model: str) -> np.ndarray:
        """Generates a dummy signal gain array for testing."""
        # The server expects 'g' from g = Hf. The size of 'g' depends on H.shape[0].
        # For a dummy, we can just create an array of a reasonable size.
        # In a real scenario, this would come from a sensor or pre-processing.
        
        # Based on typical H matrices for image reconstruction, g's length
        # could be related to the number of projections or detector elements.
        # Let's assume a fixed size for demonstration, or a size dependent on a known image size.
        
        # A common scenario for H is that H * f = g, where H is M x N, f is N x 1 (flattened image),
        # and g is M x 1 (projection data). So len(g) = M.
        # If H_1 is 10000x2500 and H_2 is 1600x400 (from original server code comments/logic implies this sort of size)
        # We need g to have len = H.shape[0]
        
        # For simplicity, let's create a signal that matches a common projection data size.
        # A 64x64 image (4096 pixels) often results in projection data of ~6000-10000 length.
        # Let's assume a size that roughly corresponds to the server's expected input for `g`.
        # The server calculates len_image = int(np.sqrt(len(resultado))), meaning `resultado` (f)
        # needs to be a perfect square for reshaping.
        # The server expects len(g) to be large for progress reporting.
        
        # Let's generate a synthetic signal gain for a 64x64 image (4096 pixels)
        # if H_1.shape[0] is say 8192
        dummy_size = 8192 # Example size, matching a common tomography measurement array size.
                         # This should ideally match `H_matrix.shape[0]` from the server's perspective.
                         # For this example, we assume server's H_1 or H_2 will handle this size for g.
        
        # Generate some random float data, scaled to be non-zero but not too large
        signal_gain = np.random.rand(dummy_size).astype(np.float64) * 100 + 50
        
        self.logger.info(f"Generated dummy signal gain of size: {signal_gain.shape[0]}")
        return signal_gain


    def _send_signal_gain(self, signal_gain: np.ndarray) -> None:
        """Sends the signal gain numpy array to the server."""
        try:
            data_bytes = signal_gain.tobytes()
            size = len(data_bytes)
            
            # Send the size of the data first (8 bytes, big-endian)
            self._send_message(f"{size}".encode('utf-8').hex()) # Send size as hex string to avoid encoding issues with direct bytes
            # The server expects `int.from_bytes(data_size, byteorder='big')`
            # So we need to send bytes directly here, not an encoded string.

            self.client_socket.sendall(size.to_bytes(8, byteorder='big'))
            self.logger.info(f"Sent signal gain size: {size} bytes.")

            # Send the actual data in chunks
            sent_bytes = 0
            while sent_bytes < size:
                chunk = data_bytes[sent_bytes : sent_bytes + self.config.BUFFER_SIZE]
                self.client_socket.sendall(chunk)
                sent_bytes += len(chunk)
                self.logger.info(f"Sent {sent_bytes}/{size} bytes of signal gain.")
            self.logger.info("Finished sending signal gain data.")
        except socket.error as e:
            self.logger.error(f"Error sending signal gain: {e}")
            self.is_connected = False
            self._handle_disconnect()
        except Exception as e:
            self.logger.error(f"Unexpected error during signal gain sending: {e}")
            self.is_connected = False
            self._handle_disconnect()


    def _reconstruct_image_request(self, option_type: int) -> None:
        """Handles the image reconstruction request flow."""
        model, image_model, algorithm_model = self._choose_reconstruction_parameters()
        
        request_message = f"OK-{self.username}-{model}-{image_model}-{algorithm_model}"
        
        self._send_message(f"OPTION-{option_type}")
        self._send_message(request_message)
        
        server_response = self._receive_message()
        if server_response.startswith("WAIT"):
            print("Server is busy. Please wait in line...")
            self.logger.info("Server reported busy, client waiting in queue.")
            # In a real application, you might want to block or provide
            # an indicator that the client is waiting. For now, we'll
            # just wait for the next message from the server which
            # will be 'OK-Pode receber' when it's their turn.
            server_response = self._receive_message() # Wait for the OK-Pode receber
            if not server_response.startswith("OK-Pode receber"):
                print("Did not receive 'OK-Pode receber' as expected after waiting.")
                self.logger.warning("Did not receive expected 'OK-Pode receber' from server.")
                time.sleep(2)
                return

        if server_response.startswith("OK-Pode receber"):
            print("Server is ready to receive signal gain data.")
            signal_gain = self._generate_dummy_signal_gain(model)
            self._send_signal_gain(signal_gain)
            
            final_response = self._receive_message()
            if final_response == "OK-Process finished":
                print("Image reconstruction completed successfully on server.")
                self.logger.info("Image reconstruction process finished on server.")
            elif final_response.startswith("ERROR"):
                print(f"Error during reconstruction: {final_response}")
                self.logger.error(f"Error from server during reconstruction: {final_response}")
            else:
                print(f"Unexpected server response after reconstruction: {final_response}")
                self.logger.warning(f"Unexpected server response after reconstruction: {final_response}")
        else:
            print(f"Server did not indicate readiness: {server_response}")
            self.logger.warning(f"Server did not indicate readiness for signal gain: {server_response}")

        time.sleep(3) # Give user time to read status

    def _calculate_file_checksum(self, filepath: str) -> str:
        """Calculates the MD5 checksum of a file."""
        checksum = hashlib.md5()
        with open(filepath, "rb") as file:
            while data := file.read(self.config.BUFFER_SIZE):
                checksum.update(data)
        return checksum.hexdigest()

    def _receive_report(self) -> None:
        """Handles the reception of a performance report file."""
        self._send_message(f"OPTION-4")
        self._send_message(f"OK-{self.username}") # Tell server we're ready for file list

        num_files_str = self._receive_message()
        try:
            num_files = int(num_files_str)
        except ValueError:
            self.logger.error(f"Invalid number of files received: {num_files_str}")
            print("Error: Invalid number of files from server.")
            return

        if num_files <= 0:
            print("No performance reports available for your username on the server.")
            self.logger.info("No performance reports available for the user.")
            self._send_message("ERROR-No files available") # Inform server we handled it
            time.sleep(2)
            return
        
        self._send_message("OK-Ready to receive file list") # Confirm readiness for file list

        print(f"\nAvailable reports ({num_files} total):")
        available_files = []
        for i in range(num_files):
            filename = self._receive_message()
            available_files.append(filename)
            print(f"{i+1}. {filename}")
            self._send_message(f"ACK-{i+1}") # Acknowledge receipt of each filename

        chosen_file = ""
        while chosen_file not in available_files:
            chosen_file = input("Enter the full filename to download: ").strip()
            if chosen_file not in available_files:
                print("Invalid filename. Please choose from the list above.")
                self._send_message(chosen_file) # Send invalid choice to server
                server_error = self._receive_message() # Expect ERROR-3-File not found!
                if server_error.startswith("ERROR"):
                    print(f"Server confirms: {server_error}")
                else:
                    print("Unexpected server response after invalid filename.")
            else:
                self._send_message(chosen_file) # Send valid choice to server
                confirmation = self._receive_message() # Expect OK-1-Confirmation
                if confirmation == "OK-1-Confirmation":
                    break
                else:
                    print(f"Server did not confirm file selection: {confirmation}")
                    self.logger.error(f"Server did not confirm file selection: {confirmation}")
                    return

        # Prepare to receive file data
        file_info_raw = self._receive_message()
        if not file_info_raw.startswith("OK-2-"):
            print(f"Error receiving file info: {file_info_raw}")
            self.logger.error(f"Error receiving file info for {chosen_file}: {file_info_raw}")
            return

        parts = file_info_raw.split("-")
        try:
            num_packets = int(parts[2])
            num_digits = int(parts[3])
            expected_checksum = parts[4]
            # print(f"Expecting {num_packets} packets for {chosen_file} with checksum {expected_checksum}")
            self.logger.info(f"Expecting {num_packets} packets for {chosen_file} with checksum {expected_checksum}")
        except (ValueError, IndexError) as e:
            self.logger.error(f"Error parsing file info from server: {file_info_raw} - {e}")
            print("Error: Could not parse file information from server.")
            return

        self._send_message("OK-Start receiving file") # Confirm readiness to receive file content

        received_data = bytearray()
        download_filepath = os.path.join(self.config.DOWNLOAD_DIR, chosen_file)

        self._clear_screen()
        self._print_title()
        print(f"Downloading '{chosen_file}'...")

        for i in range(num_packets):
            try:
                packet_data = self.client_socket.recv(self.config.BUFFER_SIZE + num_digits + 1 + 16) # Max possible size: data + index + space + hash
                if not packet_data:
                    self.logger.error(f"Server disconnected unexpectedly during packet {i+1} reception.")
                    print("Server disconnected unexpectedly.")
                    break

                # Extract index, checksum, and actual data
                try:
                    # Packet format: "index hash data"
                    first_space = packet_data.find(b' ')
                    second_space = packet_data.find(b' ', first_space + 1)
                    
                    if first_space == -1 or second_space == -1:
                        self.logger.error(f"Malformed packet received (missing spaces) for packet {i+1}.")
                        self._send_message("NOK") # Request retransmission
                        continue

                    packet_index_raw = packet_data[:first_space]
                    packet_checksum_raw = packet_data[first_space + 1 : second_space]
                    actual_data = packet_data[second_space + 1:]

                    # Verify packet integrity
                    calculated_data_checksum = hashlib.md5(actual_data).digest()
                    if calculated_data_checksum != packet_checksum_raw:
                        self.logger.warning(f"Checksum mismatch for packet {i+1}. Requesting retransmission.")
                        self._send_message("NOK")
                        continue # Skip to next iteration to re-receive
                    
                    received_data.extend(actual_data)
                    self._send_message("OK") # Acknowledge successful reception

                    current_progress = int(((i + 1) / num_packets) * 100)
                    print(f"Downloaded: {current_progress}% ({i+1}/{num_packets} packets)")

                except Exception as e:
                    self.logger.error(f"Error processing packet {i+1}: {e}")
                    self._send_message("NOK") # Request retransmission due to processing error
                    continue # Try to re-receive the same packet in next iteration of outer loop

            except socket.error as e:
                self.logger.error(f"Socket error during file download for packet {i+1}: {e}")
                self.is_connected = False
                self._handle_disconnect()
                break

        if len(received_data) > 0: # Only save if some data was received
            with open(download_filepath, "wb") as f:
                f.write(received_data)
            
            # Verify total file checksum
            final_checksum = self._calculate_file_checksum(download_filepath)
            if final_checksum == expected_checksum:
                print(f"File '{chosen_file}' downloaded successfully and verified!")
                self.logger.info(f"File '{chosen_file}' downloaded successfully and verified!")
            else:
                print(f"WARNING: File '{chosen_file}' downloaded, but checksum mismatch! Expected {expected_checksum}, got {final_checksum}")
                self.logger.warning(f"File '{chosen_file}' downloaded, but checksum mismatch!")
        else:
            print("No data received for the file.")
            self.logger.warning("No data received for the report file.")

        time.sleep(3)

    def disconnect_from_server(self) -> None:
        """Sends a disconnect request to the server."""
        if not self.is_connected:
            print("Not connected to server.")
            return

        self._send_message("OPTION-5")
        self._send_message("OK-Disconnect")
        
        response = self._receive_message()
        if response == "OK-8-Desconectado":
            print("Successfully disconnected from the server.")
            self.logger.info("Successfully disconnected from the server.")
        else:
            print(f"Unexpected response on disconnect: {response}")
            self.logger.warning(f"Unexpected response on disconnect: {response}")
        
        self._handle_disconnect()
        time.sleep(2)


    def run(self) -> None:
        """Main method to run the client."""
        while True:
            self._clear_screen()
            self._print_title()
            
            if not self.is_connected:
                connect_choice = input("1. Connect to Server\n2. Exit\nEnter choice: ").strip()
                if connect_choice == '1':
                    if not self.connect_to_server():
                        continue
                elif connect_choice == '2':
                    self.logger.info("Client application exited.")
                    break
                else:
                    print("Invalid choice.")
                    time.sleep(1)
                    continue
            
            if self.is_connected:
                choice = self.display_options()
                if choice == '1':
                    self._reconstruct_image_request(1)
                elif choice == '2':
                    self._reconstruct_image_request(2)
                elif choice == '3':
                    self._reconstruct_image_request(3)
                elif choice == '4':
                    self._receive_report()
                elif choice == '5':
                    self.disconnect_from_server()
            else:
                print("Not connected to server. Returning to main menu.")
                time.sleep(1)


if __name__ == "__main__":
    client = Client()
    client.run()