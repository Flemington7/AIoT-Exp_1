import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from load import *


class ClientsGroup(object):
    def __init__(self, device, clients_num):
        self.device = device
        self.clients_set = {}
        self.clients_num = clients_num
        self.dataset_balance_allocate()  # initialize the clients_set

    def dataset_balance_allocate(self):
        arranged_index = getdata(self.clients_num)

        for client_index, data_index in enumerate(arranged_index):  # data_index is a list
            local_data_num = len(data_index)
            local_label, local_feature = (np.vstack(all_labels_train[data_index]),
                                          np.vstack(all_features_train[data_index]))
            client_object = Client(TensorDataset(torch.tensor(local_feature, dtype = torch.float, requires_grad = True),
                                                 torch.tensor(local_label, dtype = torch.float, requires_grad = True)),
                                   self.device, local_data_num)

            self.clients_set['client{}'.format(client_index + 1)] = client_object


class Client(object):
    def __init__(self, local_train_dataset, device, local_data_num):
        self.train_dataset = local_train_dataset
        self.device = device
        self.train_dataloader = None
        self.data_num = local_data_num
        self.state = {}

    def local_update(self, local_batch_size, local_epoch, net, loss_func, opti, global_parameters):
        net.load_state_dict(global_parameters, strict = True)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size = local_batch_size, shuffle = True)
        for epoch in range(local_epoch):
            for features, labels in self.train_dataloader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                predict = net(features)
                loss = loss_func(predict, labels)
                loss.backward()
                opti.step()
                opti.zero_grad()
        return net.state_dict()
