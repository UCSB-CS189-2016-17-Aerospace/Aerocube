from tcpServer import TcpServer



def main():
	server = TcpServer('127.0.0.1',5005,1024)
	server.accept_connection()
	while 1:
		data = server.receive_data()
		if data != False:
			print(data)
			response = { 'words' : 'go here'}
			server.send_response(response)




if __name__ == '__main__':
	main()