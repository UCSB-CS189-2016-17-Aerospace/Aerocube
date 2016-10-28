import subprocess 

pwd_output = subprocess.check_output(['pwd'])

print(pwd_output)