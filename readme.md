environment configuration:
Python 3.11
conda install pandas
conda install numpy
conda install scikit-learn
conda install matplotlib
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 #不小心在base也安装了
pip install tqdm

parameter configuration：
客户端数量number of clients = 10 (so that each client has 300 data)
全局轮global epoch = 10
本地轮local epoch = 5
device = RTX 4060 mobile

net:
batchsize = 64
Linear1 13 * 16
Linear2 16 * 1
activation function = sigmoid
optimizer = SGD
loss function = MSEloss reduction = 'mean'
learning rate = 0.01
