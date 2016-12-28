echo "Curling /home/ubuntu/GitHub/Aerocube/ImP/imageProcessing/test_files/jetson_test1.jpg"
curl -F photo=@/home/ubuntu/GitHub/Aerocube/ImP/imageProcessing/test_files/jetson_test1.jpg https://localhost:3000/api/uploadImage -k
# echo "/home/ubuntu/GitHub/Aerocube/flaskServer/static/img/epicEarthMoon.png"
# curl -F photo=@/home/ubuntu/GitHub/Aerocube/flaskServer/static/img/epicEarthMoon.png https://localhost:3000/api/uploadImage -k
