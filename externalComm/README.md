need to pip install pyrebase.

to use import process from externalComm.
process(func, database, scanID, location=None, data=None, testing=False)
    func: '-w'|'-r'|'-d'|'-iw'|'-dl'
    database: 'firebase'
    location: path in database
    scanID: id of scan
    data:  data
    testing: if testing true else leave alone
and will carry out command.  
-dl does not work
getting `{
  "error": {
    "code": 403,
    "message": "Permission denied. Could not perform this operation"
  }
}`