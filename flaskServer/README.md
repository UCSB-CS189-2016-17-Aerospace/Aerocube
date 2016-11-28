#Flask Server

This part of the documentation covers all the interfaces of the Flask Server:

##/api/uploadImage  
**Method:**
GET  
**Params:**  
None

**Success Response**
```
Code: 200    
Content: {'server status': 'server is up and running'}
```

##/api/uploadImage  
**Method:**
POST  
**Params:**  
photo = [string of file name] _required_

**Success Response**
```
Code: 200
Content: {'upload status' : 'file upload sucessful'}
```

**Sample Call:**
```
$ curl -F photo=@/Users/gustavo/Desktop/pictureForUpload.png http://127.0.0.1:5000/api/uploadImage
```
