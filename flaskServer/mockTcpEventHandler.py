from tcpClient import TcpClient 

def main():
	client = TcpClient('127.0.0.1',5005,1024)
	client.connect_to_controller()
	message = { 'words' : 'more words'}
	client.send_to_controller(message)
	data = client.receive_from_controller()
	print(data)
	client.close()



if __name__ == '__main__':
	main()