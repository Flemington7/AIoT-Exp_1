import argparse
import time

import torch.cuda
from torch import optim
from tqdm import tqdm

from clients import ClientsGroup
from model import Weather_2NN

parser = argparse.ArgumentParser(description = 'FedAvg')
parser.add_argument('--num_of_clients', type = int, default = 10,
                    help = 'number of clients (default: 10)')
parser.add_argument('--global_epoch', type = int, default = 10,
                    help = 'number of global epochs to train (default: 10)')
parser.add_argument('--local_epoch', type = int, default = 5,
                    help = 'number of local epochs to train (default: 5)')
parser.add_argument('--local_batch_size', type = int, default = 64,
                    help = 'local batch size (default: 64)')
parser.add_argument('--learning_rate', type = float, default = 0.05,
                    help = 'learning rate (default: 0.05)')
args = parser.parse_args().__dict__

net = Weather_2NN()
'''if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = torch.nn.DataParallel(net)'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)
loss_func = torch.nn.MSELoss(reduction = 'mean')
opti = optim.SGD(net.parameters(), lr = args['learning_rate'])
global_parameters = net.state_dict()  # get the parameters of the model

Clients = ClientsGroup(device, args['num_of_clients'])
clients_sequence = ['client{}'.format(i + 1) for i in range(Clients.clients_num)]
weigh = 1 / Clients.clients_num

print('--------------------------------FedAvg-----------------------------------')
for i in range(1, args['global_epoch'] + 1):
    print('----------------------------global epoch', i, '------------------------------')
    time.sleep(0.1)
    sum_parameters = None
    for client in tqdm(clients_sequence):
        local_parameters = Clients.clients_set[client].local_update(args['local_batch_size'], args['local_epoch'], net,
                                                                    loss_func, opti, global_parameters)
        if sum_parameters is None:
            sum_parameters = local_parameters
        else:
            for var in sum_parameters:
                sum_parameters[var] = sum_parameters[var] + local_parameters[var]
    # update the global parameters
    for var in global_parameters:
        global_parameters[var] = sum_parameters[var] * weigh
