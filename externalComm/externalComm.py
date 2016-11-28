import json
from .commClass import FirebaseComm
#TODO: data_json finalization
def process(func, database, scanID, location=None, data=None, testing=False):
    """
    :param func: '-w'|'-r'|'-d'|'-iw' |'-dl'
    :param database: 'firebase'
    :param location: path in database
    :param scanID: id of scan
    :param data:  data
    :param testing: if testing true else leave alone
    :return:
    """
    comm=None
    if(database=='firebase'):
        comm=FirebaseComm(testing)
    else:
        raise ValueError('database not found!')
    if func == '-w':
        comm.write(location=location, id=scanID, data=data)
    elif func == '-r':
        return comm.read(location=location, id=scanID)
    elif func == '-d':
        comm.delete(location=location, id=scanID)
    elif func == '-iw':
        comm.imageStore(id=scanID, srcImage=data)
    elif func == '-dl':
        comm.imageDownload(id=scanID)
    else:
        raise SyntaxError('func not accepted, -w for write, -r for read.')
