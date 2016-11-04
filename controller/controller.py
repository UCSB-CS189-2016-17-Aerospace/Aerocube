import sys
from controller.tcpServer import TcpServer
# this is how we do it if we were in the same repo
from externalComm.externalComm import process
from dataStorage.dataStrorage import store,read
sys.path.insert(1, 'home/ubuntu/Github/Aerocube-ImP')
from fiducialMarkerModule.fiducialMarker import fiducialMarker, \
 IDOutOfDictionaryBoundError
# DEAL WITH IMPORTS
# look into 
# import os
# os.chdir(default_path)

class Controller:
	def __init__(self):
		self.server = TcpServer('127.0.0.1',5005,1024)

	def return_status(self, status):
		'''
		returns status back to event handler
		:param status: status signal
		:return: void
		'''
		result_event_status=ResultEvent(result_signal=status)
		self.server.send_response(result_event_status)

	def scan_image(self, file_path):
		imp = ImageProcessor(file_path)
		return imp._find_fiducial_markers() #assuming this method returns the vectors and corners

	def store_locally(self, path, data):
        dataStorage.store(location=path,pickleable=data)

	def store_data_externally(self, database, ID, data):
		process(func='-w',database=database,scanID=ID,data=data)

    def initiate_scan(self,scan_ID,payload):
        results = self.scan_image(payload.string(0))##payload.string(0) should be the path to the image
        self.store_locally(path=scan_ID,data=results)
        self.store_data_externally(database=payload.string(1),ID=scan_ID,data=results)#payload.string(1) should be the database
        self.return_status()


	def run():
		self.server.accept_connection()
		while 1:
			data = self.server.receive_data()
			if data != False:
				if data.signal() == ImageEventSignal.IDENTIFY_AEROCUBES:
					self.initiate_scan(scan_ID=data.created_at, payload=data.payload())
				else:
					pass
					#IM CONFUSED AF

if __name__ == '__main__':
	controller = Controller()
	controller.run()
