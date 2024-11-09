import torch
import os
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
def plott(observation):
    plt.figure()
    plt.plot(observation[0, :, 0], 'red')
    plt.plot(observation[1, :, 0], 'blue')
    plt.plot(observation[0, :, 1], 'green')
    plt.plot(observation[1, :, 1], 'purple')

class StateEncoder(nn.Module):
    def __init__(self, out_dim=3):
        super(StateEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, out_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(out_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )

        for ly in self.encoder:
            if ly is nn.Linear:
                nn.init.uniform_(ly.weight.data, -c, c)
                nn.init.constant_(ly.bias.data, 0)

        for ly in self.decoder:
            if ly is nn.Linear:
                nn.init.uniform_(ly.weight.data, -c, c)
                nn.init.constant_(ly.bias.data, 0)


    def forward(self, s, dim):
        # s = s.flatten(start_dim=1)
        if dim == 0:
            s = s[:, :, :, 0].flatten(end_dim=1)
        else:
            s = s[:, :, :, 1].flatten(end_dim=1)
        temp = self.encoder(s)
        return self.decoder(temp), temp, s


class StateActionMapping(nn.Module):
    def __init__(self, d, num_groups, embedding_size=64, input_size=130):
        super(StateActionMapping, self).__init__()
        self.d = d
        self.num_groups = num_groups

        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, embedding_size)
        self.ReLU = nn.ReLU()
        self.Softmax = nn.Softmax(dim=1)

        self.lr_r = nn.Sequential(
            nn.Linear(embedding_size, 1, bias=False),
            nn.Sigmoid()
        )

        self.lr_g = nn.Sequential(
            nn.Linear(embedding_size, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, s, a, model=None):
        s = s.transpose(2, 3).reshape(-1, 4, 32)
        # s[:, :, 1:] = torch.diff(s, dim=2)
        s = s.flatten(start_dim=1)
        s = (s - 0.1)
        t = torch.cat([s, a], dim=1) * 10.0

        e1 = self.fc1(t)
        e2 = self.ReLU(e1)
        e3 = self.fc2(e2)
        e4 = self.ReLU(e3)
        e5 = self.fc3(e4)
        e6 = self.ReLU(e5)
        e7 = self.fc4(e6)
        phi = self.Softmax(e7)
        r_p = self.lr_r(phi)
        g_p = self.lr_g(phi)
        # g_p = 0
        return phi, r_p, g_p
    # def __init__(self, d, num_groups, embedding_size, input_size=128):
    #     super(StateActionMapping, self).__init__()
    #     self.d = d
    #     self.num_groups = num_groups
    #
    #     # self.state_encoder = nn.Sequential(
    #     #     nn.Linear(input_size, 400),
    #     #     # nn.Sigmoid(),
    #     #     nn.ReLU(),
    #     #     nn.Linear(400, 300),
    #     #     # nn.Sigmoid(),
    #     #     nn.ReLU(),
    #     #     nn.Linear(300, embedding_size),
    #     #     nn.Softmax(dim=1)
    #     # )
    #
    #     self.fc1 = nn.Linear(input_size, 400)
    #     self.fc2 = nn.Linear(400, 300)
    #     self.fc3 = nn.Linear(300, embedding_size)
    #     self.ReLU = nn.ReLU()
    #     self.Softmax = nn.Softmax(dim=1)
    #
    #     self.encoder = nn.Sequential(
    #         nn.Linear(embedding_size + num_groups, 32),
    #         nn.ReLU(),
    #         # nn.Sigmoid(),
    #         nn.Linear(32, 32),
    #         nn.ReLU(),
    #         # nn.Sigmoid(),
    #         nn.Linear(32, self.d),
    #         nn.Softmax(dim=1)
    #     )
    #
    #     self.lr_r = nn.Sequential(
    #         nn.Linear(self.d, 1, bias=False),
    #         nn.Sigmoid()
    #     )
    #
    #     self.lr_g = nn.Sequential(
    #         nn.Linear(self.d, 1, bias=False),
    #         nn.Sigmoid()
    #     )
    #
    #     # for ly in self.encoder:
    #     #     if ly is nn.Linear:
    #     #         nn.init.uniform_(ly.weight.data, -c, c)
    #     #         nn.init.constant_(ly.bias.data, 0)
    #     # for ly in self.state_encoder:
    #     #     if ly is nn.Linear:
    #     #         nn.init.uniform_(ly.weight.data, -c, c)
    #     #         nn.init.constant_(ly.bias.data, 0)
    #     # for ly in self.lr_g:
    #     #     if ly is nn.Linear:
    #     #         nn.init.uniform_(ly.weight.data, -c, c)
    #     #         nn.init.constant_(ly.bias.data, 0)
    #     #
    #     # for ly in self.lr_r:
    #     #     if ly is nn.Linear:
    #     #         nn.init.uniform_(ly.weight.data, -c, c)
    #     #         nn.init.constant_(ly.bias.data, 0)
    #
    #
    # def forward(self, s, a, model=None):
    #     s = s.transpose(2, 3).reshape(-1, 4, 32)
    #     # s[:, :, 1:] = torch.diff(s, dim=2)
    #     s = s.flatten(start_dim=1)
    #     s = (s - 0.1) * 10
    #
    #     e1 = self.fc1(s)
    #     e2 = self.ReLU(e1)
    #     e3 = self.fc2(e2)
    #     e4 = self.ReLU(e3)
    #     e5 = self.fc3(e4)
    #     e = self.Softmax(e5)
    #
    #
    #     # e = self.state_encoder(s)
    #     # a = a.reshape(-1, self.num_groups, 1)
    #     t = torch.cat([e, a], dim=1)
    #     # t = t.flatten(start_dim=1)
    #     phi = self.encoder(t)
    #     r_p = self.lr_r(phi)
    #     g_p = self.lr_g(phi)
    #     # g_p = 0
    #     return phi, r_p, g_p
    #
    # def __init__(self, d, num_groups, embedding_size):
    #     super(StateActionMapping, self).__init__()
    #     self.d = d
    #     self.num_groups = num_groups
    #     self.encoder = nn.Sequential(
    #         nn.Linear(embedding_size + num_groups, 32),
    #         nn.ReLU(),
    #         nn.Linear(32, 32),
    #         nn.ReLU(),
    #         nn.Linear(32, self.d),
    #         nn.Softmax(dim=1)
    #     )
    #     for ly in self.encoder:
    #         if ly is nn.Linear:
    #             nn.init.uniform_(ly.weight.data, -c, c)
    #             nn.init.constant_(ly.bias.data, 0)
    #     self.lr_r = nn.Sequential(
    #         nn.Linear(self.d, 1, bias=False),
    #         nn.Sigmoid()
    #     )
    #
    #     self.lr_g = nn.Sequential(
    #         nn.Linear(self.d, 1, bias=False),
    #         nn.Sigmoid()
    #     )
    #
    #     # self.lr_r = nn.Sequential(
    #     #     nn.Linear(self.d, 2),
    #     #     nn.Softmax(dim=1)
    #     # )
    #     #
    #     # self.lr_g = nn.Sequential(
    #     #     nn.Linear(self.d, 2),
    #     #     nn.Softmax(dim=1)
    #     # )
    #
    # def forward(self, s, a, model):
    #     with torch.no_grad():
    #         e = model.encode(s)
    #     a = a.reshape(-1, self.num_groups, 1)
    #     t = torch.cat([e, a], dim=2)
    #     t = t.flatten(start_dim=1)
    #     phi = self.encoder(t)
    #     r_p = self.lr_r(phi)
    #     g_p = self.lr_g(phi)
    #     # g_p = 0
    #     return phi, r_p, g_p

    # def compute(self, state, action, multi=False):
    #     if multi:
    #         num = action.shape[0]
    #         return torch.ones((num, self.d), device=self.device)
    #     else:
    #         return torch.ones((self.d,), device=self.device)
#
# def transform(s):
#     temp = s[:, :, :, :2]
#     temp[:, :, 1:, 1] = torch.diff(temp[:, :, :, 1], dim=2)
#     temp[:, :, 1:, 0] = torch.diff(temp[:, :, :, 0], dim=2)
#     # temp = F.normalize(temp, dim=2)
#     # temp[:, :, :, 0] /= torch.linalg.norm(temp[:, :, :, 0])
#     # temp[:, :, :, 1] /= torch.linalg.norm(temp[:, :, :, 1])
#     temp /= torch.max(temp)
#     return temp

def train_vae(dim=0):
    bs = 32
    d = 20
    c = 1
    epochs = 100
    lr = 0.031
    device = 'cpu'

    train_data = torch.load('./data/train_phi.pt')
    s = train_data['s']
    a = train_data['a']
    r = train_data['r']
    g = train_data['g']
    # s = transform(s)

    dataset = TensorDataset(s, a, r, g)
    s_size = s.size()[1]
    loader = DataLoader(dataset, batch_size=bs)
    model = StateEncoder().to(device)
    optim = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )
    criterion = nn.MSELoss().to(device)
    for epoch in range(epochs):
        for s, a, r, g in loader:
            s_p, temp, s_o = model(s, dim=dim)
            loss = criterion(s_p, s_o)
            loss.backward()
            optim.step()
            optim.zero_grad()
        print(epoch, loss)
        # print(temp[:5])

    torch.save(model.state_dict(), './data/state_encoder_'+str(dim)+'.pt')
def train():
    bs = 128
    d = 10
    c2 = 0.0
    c1 = 1 - c2
    epochs = 100000
    lr = 0.005
    device = 'cpu'
    train_data = torch.load('./data/train_phi.pt')
    s = train_data['s']
    a = train_data['a']
    r = train_data['r']
    g = train_data['g']
    # s = transform(s)
    # s_encoder = StateEmbedding()
    # test_s, test_a = s[0].unsqueeze(0), a[0].unsqueeze(0)

    l = s.shape[0]


    losses = []

    dataset = TensorDataset(s, a, r, g)
    loader = DataLoader(dataset, batch_size=bs)

    model = StateActionMapping(d, 2, 64, 130).to(device)

    optim = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )
    # criterion = nn.CrossEntropyLoss().to(device)
    criterion = nn.MSELoss().to(device)
    # print(model.state_dict()['lr_g.weight'])
    for epoch in range(epochs):
        if epoch > 0 and epoch % 100 == 0:
            torch.save(model.state_dict(), './data/phi_model.pt')
            L = torch.hstack(losses)
            torch.save(L, './data/losses.pt')
        # print(model(test_s, test_a, s_encoder)[0])
        for s, a, r, g in loader:
            phi, r_p, g_p = model(s, a)
            # loss = criterion(r_p.flatten(), r)
            # loss = criterion(g_p.flatten(), g)
            loss = c1 * criterion(r_p.flatten(), r) + c2 * criterion(g_p.flatten(), g)
            # print(loss)
            loss.backward()
            optim.step()
            optim.zero_grad()
        print(epoch, loss)
        losses.append(loss)
    # torch.save(model.state_dict(), 'phi_model.pt')
    # L = torch.hstack(losses)
    # torch.save(L, 'losses.pt')

# def train():
#     bs = 32
#     d = 10
#     c1 = 1
#     c2 = 1
#     epochs = 100000
#     lr = 1e-3
#     device = 'cpu'
#     train_data = torch.load('./data/train_phi.pt')
#     s = train_data['s']
#     a = train_data['a']
#     r = train_data['r']
#     g = train_data['g']
#     # s = transform(s)
#     s_encoder = StateEmbedding()
#     test_s, test_a = s[0].unsqueeze(0), a[0].unsqueeze(0)
#
#
#
#     losses = []
#
#     dataset = TensorDataset(s, a, r, g)
#     loader = DataLoader(dataset, batch_size=bs)
#
#     model = StateActionMapping(d, 2, 12).to(device)
#     optim = torch.optim.Adam(
#         model.parameters(),
#         lr=lr,
#         # weight_decay=1e-4,
#     )
#     # criterion = nn.CrossEntropyLoss().to(device)
#     criterion = nn.MSELoss().to(device)
#     # print(model.state_dict()['lr_g.weight'])
#     for epoch in range(epochs):
#         if epoch > 0 and epoch % 100 == 0:
#             torch.save(model.state_dict(), './data/phi_model.pt')
#             L = torch.hstack(losses)
#             torch.save(L, './data/losses.pt')
#         # print(model(test_s, test_a, s_encoder)[0])
#         for s, a, r, g in loader:
#             # print(model.state_dict()['lr_g.weight'])
#             # with torch.no_grad():
#                 # print(model(test_s, test_a, s_encoder)[0])
#             #
#             # s_encoder.encoder0.load_state_dict(torch.load('state_encoder_0.pt', map_location=device))
#             # s_encoder.encoder1.load_state_dict(torch.load('state_encoder_1.pt', map_location=device))
#
#
#             # print(s_encoder.encode(test_s))
#             # r_t = torch.stack((1 - r, r)).T
#             # g_t = torch.stack((1 - g, g)).T
#             phi, r_p, g_p = model(s, a, s_encoder)
#             # loss = criterion(r_p.flatten(), r)
#             # loss = criterion(g_p.flatten(), g)
#             loss = c1 * criterion(r_p.flatten(), r) + c2 * criterion(g_p.flatten(), g)
#             loss.backward()
#             optim.step()
#             optim.zero_grad()
#         print(epoch, loss)
#         losses.append(loss)
#     # torch.save(model.state_dict(), 'phi_model.pt')
#     # L = torch.hstack(losses)
#     # torch.save(L, 'losses.pt')

class StateEmbedding:
    def __init__(self, dim=3, device='cpu'):
        self.embed_dim = dim
        self.device = device
        self.encoder0 = StateEncoder(out_dim=3)
        self.encoder1 = StateEncoder(out_dim=3)
        path = os.getcwd()
        self.encoder0.load_state_dict(torch.load(path + './data/state_encoder_0.pt', map_location=self.device))
        self.encoder1.load_state_dict(torch.load(path + './data/state_encoder_1.pt', map_location=self.device))
        self.encoder0.eval()
        self.encoder1.eval()

    def encode(self, s):
        with torch.no_grad():
            s = transform(s)
            _, e0, _ = self.encoder0(s, dim=0)
            _, e1, _ = self.encoder1(s, dim=1)
            return torch.cat([e0, e1], dim=1).reshape(-1, 2, self.embed_dim*2)

def test_encoder():
    train_data = torch.load('train_phi.pt')
    s = train_data['s']
    a = train_data['a']
    r = train_data['r']
    g = train_data['g']

    c = StateEmbedding()

def weights(model1, model2):
    for key in model1.state_dict():
        if torch.sum(model2.state_dict()[key] - model1.state_dict()[key]) > 1e-5:
            return False
    return True



# train_vae(dim=0)
# train_vae(dim=1)
train()
# test_encoder()
