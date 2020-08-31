echo "####### PYTHON3 #######"
sudo apt-get install python3

echo "####### MATPLOTLIB #######"
git clone https://github.com/matplotlib/matplotlib
cd matplotlib
python3 setup.py build 
sudo python3 setup.py install



echo "####### SCIPY ########"
sudo apt update 
sudo apt install -y python3-scipy 


echo "####### IPYTHON #######"
pip3 install ipython


echo "####### PIL #########"
sudo apt-get install -y python3-dev python3-setuptools
sudo apt-get install -y python3-setuptools



echo "####### PYNPUT ########"
pip3 install pynput

echo "######## DLIB #########"
sudo apt-get update
sudo apt-get install -y build-essential cmake
sudo apt-get install -y libopenblas-dev liblapack-dev libatlas-base-dev
sudo apt-get install -y libx11-dev libgtk-3-dev


mkvirtualenv dlib_test -p python
workon cv


pip3 install numpy
pip3 install  dlib

git clone https://github.com/davisking/dlib.git
cd dlib

python3 setup.py install --yes USE_NEON_INSTRUCTIONS


echo "####### OPENCV ########"
sudo rpi-update

sudo apt-get install -y build-essential cmake pkg-config
sudo apt-get install -y libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install -y libxvidcore-dev libx264-dev
sudo apt-get install -y libgtk2.0-dev libgtk-3-dev
sudo apt-get install -y libatlas-base-dev gfortran

sudo apt-get install -y python3 python3-setuptools python3-dev

wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py

# cd ~
# wget -O opencv.zip https://github.com/Itseez/opencv/archive/3.4.1.zip
# wget -O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/3.4.1.zip
# unzip opencv.zip
# unzip opencv_contrib.zip

sudo pip3 install numpy

pip3 install opencv-python
