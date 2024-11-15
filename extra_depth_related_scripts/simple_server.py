import socket

# Define the port number
PORT = 21999  # Change this to the desired port

def start_server():
    # Create a TCP/IP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to all network interfaces and the specified port
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(("0.0.0.0", PORT))

    # Listen for incoming connections
    server_socket.listen(5)  # Allows up to 5 connections in queue
    print(f"Server listening on port {PORT}")

    while True:
        # Wait for a connection
        client_socket, client_address = server_socket.accept()
        print(f"Connection received from {client_address}")

        # Send a simple response and close the connection
        client_socket.sendall(b"Hello from server")
        client_socket.close()

if __name__ == "__main__":
    start_server()
