import sys
from tcpServer import TcpServer
sys.path.insert(1, 'home/ubuntu/Github/Aerocube-ImP')
from fiducialMarkerModule.fiducialMarker import fiducialMarker, \
 IDOutOfDictionaryBoundError
# DEAL WITH IMPORTS

class Controller:
	def __init__(self):
		self.server = TcpServer('127.0.0.1',5005,1024)

	def return_status(self, status):
		#either already recieve a type signal status or make it here
		self.server.send_response(result_event_status)

	def scan_image(self, file_path):
		imp = ImageProcessor(file_path)
		return imp._find_fiducial_markers() #assuming this method returns the vectors and corners

	def store_locally(self, path, data):
		pass

	def store_externally(self, database, data):
		pass

	def initiate_scan():
		results = self.scan_image()
		self.store_locally('path',results)
		self.store_externally('database',results)
		self.return_status()

	def run():
		self.server.accept_connection()
		while 1:
			data = self.server.receive_data()
			if data != False:
				if data.signal() == ImageEventSignal.IDENTIFY_AEROCUBES:
					self.initiate_scan()
				else:
					pass
					#IM CONFUSED AF

if __name__ == '__main__':
	controller = Controller()
	controller.run()
