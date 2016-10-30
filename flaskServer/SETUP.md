# Setting up the Flask Server and Development Environment 

### Install PIP

PIP is a package management system that we will use to install just about everything for python/Flask.
```shell
sudo apt-get install python-pip
```

### Install Virtual Environment 

Virtual Environment is a tool to help keep dependencies required by different projects in different places. It creates a python virtual environment for each project where you can then install any additional libraries, away from other projects.

#####Using the newly installed PIP, install virtual environment:
```shell
pip install virtualenv
```

After installing virtual environment, cd into your project directory and create the environemnt (I named it venv, feel free to call it whatever you want):
```shell
virtualenv venv
```

This will create a folder within your project directory that holds the python environment.

Once this is set, you can begin to work by activating the environment in your project directory:
```shell
. venv/bin/activate
```
When you want to leave that particular environment, you simply close it with this command:
```shell
deactivate
```

### Requirements

Thanks to PIP and virtualenv, when we need to install new packages or libraries, simply activate your environment and PIP away:
```shell
pip install Flask
pip install flask-restful
```

All of the requirements are on the requirements.txt file so you can just pip install that.
```shell
pip install -r requirements.txt
```

### To Run the Flask server
cd into the flask directory

```shell
# Sample run
python restSample.py

# Sample test run
python testRest.py
```

### Interacting with the uploadImage endpoint via commandline 
 You can interact with any endpoint by using CURL:
 ```shell
 curl -F photo=@/Users/gustavo/Desktop/pictureForUpload.png http://127.0.0.1:5000/api/uploadImage
 ```
 Just make sure to set your own path for the image you are uploading. 
