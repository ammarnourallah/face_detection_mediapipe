# Python/C++ implementation of Face Detection of mediapipe

## Installation
pip install opencv-python tensorflow
### edgetpu installation
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

sudo apt-get update
sudo apt-get install libedgetpu1-std
sudo apt-get update

### install tflite-runtime for rpi4
pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl

python main.py

### Requirements

* Python 3.5+
* Linux, Windows or macOS, Raspberry pi4

