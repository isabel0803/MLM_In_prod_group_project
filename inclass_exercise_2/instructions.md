# Commands to run

docker build -t qr-generator . 

docker run --rm -v "$(pwd)":/app qr-generator -o my_qr.png "https://facebook.com"  


