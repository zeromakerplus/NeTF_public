# NeTF_public
The re is the released code for paper "Non-line-of-Sight Imaging via Neural Transient Fields".

requirements is going to be updated

The preprocessed data we use can be downloaded at [Google Drive](https://drive.google.com/file/d/1kGVrFcNZZbZs0ute_roEOg5UkYeh3jRl/view?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/16lWXwhm8CbXWAumJmlw9MQ) with password: netf

The raw data can be downloaded at [Zaragoza NLOS synthetic dataset](https://graphics.unizar.es/nlos_dataset.html), [f-k migration](http://www.computationalimaging.org/publications/nlos-fk/) and [Convolutional Approximations](https://imaging.cs.cmu.edu/conv_nlos/)

# environment setup
"pip install -r requirements.txt"


# How to run
"python run_netf.py --config configs/zaragoza_bunny.txt"
preset settings are prepared for different scenes at "./configs/".
