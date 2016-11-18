# External Commmunication Module
## Setup
Need to
```
pip install pyrebase.
```

## Usage
To use, must import first:
```
import process from externalComm.
```
Function Signature
```
process(func, database, scanID, location=None, data=None, testing=False)
    func: '-w'|'-r'|'-d'|'-iw'|'-dl'
    database: 'firebase'
    location: path in database
    scanID: id of scan
    data:  data
    testing: if testing true else leave alone
```
and will carry out command.  
Note that flag '-dl' does not work. The following error is returned:
```
`{
  "error": {
    "code": 403,
    "message": "Permission denied. Could not perform this operation"
  }
}`
```
