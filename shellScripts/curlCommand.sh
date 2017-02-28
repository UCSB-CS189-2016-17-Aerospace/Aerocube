#!/usr/bin/env bash
#IMG1=/home/$USER/GitHub/Aerocube/ImP/imageProcessing/test_files/jetson_test1.jpg
IMG1=/home/$USER/GitHub/Aerocube/ImP/imageProcessing/test_files/capstone_class_photoshoot/SPACE_1.JPG
# IMG1=/home/$USER/GitHub/Aerocube/ImP/imageProcessing/test_files/absol.jpg
echo "Curling $IMG1"
curl -F photo=@${IMG1} https://localhost:3000/api/uploadImage -k
# echo "/home/ubuntu/GitHub/Aerocube/flaskServer/static/img/epicEarthMoon.png"
# curl -F photo=@/home/ubuntu/GitHub/Aerocube/flaskServer/static/img/epicEarthMoon.png https://localhost:3000/api/uploadImage -k
