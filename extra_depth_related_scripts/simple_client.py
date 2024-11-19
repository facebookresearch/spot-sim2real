import socket
import os

SPOT_IP = os.environ["SPOT_IP"]
# Define the IP and port of the server
SERVER_IP = SPOT_IP  # Replace with your NUC's IP
PORT = 21999               # Port to match the server's port

def ping_server():
    # Create a TCP/IP socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        # Attempt to connect to the server
        client_socket.connect((SERVER_IP, PORT))
        print("Connected to server successfully.")
        
        # Receive a response
        response = client_socket.recv(1024)
        print("Received from server:", response.decode())
    
    except socket.error as e:
        print(f"Connection to server failed: {e}")
    
    finally:
        client_socket.close()

if __name__ == "__main__":
    ping_server()
