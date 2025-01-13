# PADC: Predictive-and-Adaptive-Deep-Coding-for-Wireless-Image-Transmission-in-Semantic-Communication
Pytorch code for IEEE TWC paper "Predictive and Adaptive Deep Coding for Wireless Image Transmission in Semantic Communication"

For more details, please read to the following paper: 
Zhang W, Zhang H, Ma H, et al. Predictive and Adaptive Deep Coding for Wireless Image Transmission in Semantic Communication. IEEE Transactions on Wireless Communications, 2023. 


# 0. Environment
```bash
conda create -n padc python=3.8
conda activate padc
pip install -r requirements.txt
```

# 1. Data Preprocessing
```bash
# CIFAR10, CIFAR100
python download.py
```