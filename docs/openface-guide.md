# How to install OpenFace 

To install OpenFace in the root directory of this project:
```
git clone https://github.com/TadasBaltrusaitis/OpenFace.git
cd OpenFace
```
Then download the needed models by:
```
bash download_models.sh
```
It's better to install some dependencies required by OpenCV before you run 'install.sh':
```
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install build-essential cmake pkg-config # Install developer tools used to compile OpenCV
sudo apt-get install libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev #  Install libraries and packages used to read various image formats from disk
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libxvidcore-dev libx264-dev # Install a few libraries used to read video formats from disk
```
And then install OpenFace by:
```
bash install.sh
```


