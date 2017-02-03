#!/usr/bin/env bash
IMG1=/home/andrew/GitHub/Aerocube/ImP/imageProcessing/test_files/jetson_test1.jpg
echo "Curling $IMG1"
curl -F photo=@${IMG1} https://localhost:3000/api/uploadImage -k
# echo "/home/ubuntu/GitHub/Aerocube/flaskServer/static/img/epicEarthMoon.png"
# curl -F photo=@/home/ubuntu/GitHub/Aerocube/flaskServer/static/img/epicEarthMoon.png https://localhost:3000/api/uploadImage -k
