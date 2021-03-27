# NeTF_public
The repository is the released code for paper "Non-line-of-Sight Imaging via Neural Transient Fields". [[paper]](https://arxiv.org/abs/2101.00373#:~:text=Title%3ANon-line-of-Sight%20Imaging%20via%20Neural%20Transient%20Fields.%20Non-line-of-Sight%20Imaging,within%20a%20pre-defined%20volume%29%20of%20the%20hidden%20scene.)

The preprocessed data we use can be downloaded at [[Google Drive]](https://drive.google.com/file/d/1kGVrFcNZZbZs0ute_roEOg5UkYeh3jRl/view?usp=sharing) or [[Baidu Netdisk]](https://pan.baidu.com/s/16lWXwhm8CbXWAumJmlw9MQ) with password: netf

The raw data can be downloaded at [Zaragoza NLOS synthetic dataset](https://graphics.unizar.es/nlos_dataset.html), [f-k migration](http://www.computationalimaging.org/publications/nlos-fk/) and [Convolutional Approximations](https://imaging.cs.cmu.edu/conv_nlos/)

# environment setup
"pip install -r requirements.txt"


# How to run
"python run_netf.py --config configs/zaragoza_bunny.txt"
preset settings are prepared for different scenes at "./configs/".
