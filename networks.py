"""
Houses networks definitions
"""

import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable

class OmniglotModel(nn.Module):
    def __init__(self, num_filters=64, embed_dim=64, use_cuda=False):
        super(OmniglotModel, self).__init__()
        self.use_cuda = use_cuda
        self.num_filters = num_filters
        # encoder params
        self.c1 = nn.Conv2d(1, num_filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)

        self.c2 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

        self.c3 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_filters)

        self.c4 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(num_filters)

        self.l_mu = nn.Linear(num_filters, embed_dim)
        self.l_log_var = nn.Linear(num_filters, embed_dim)

        # decoder params
        self.dl1 = nn.Linear(embed_dim, 2 * 2 * num_filters)
        self.dbn1 = nn.BatchNorm1d(2 * 2 * num_filters)
        #self.unpool1 = nn.ConvTranspose2d(num_filters, num_filters, 2, 2, bias=False)
        #self.unpoolbn1 = nn.BatchNorm2d(num_filters)

        self.dc2 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        #self.dc2 = nn.ConvTranspose2d(num_filters, num_filters, 3, 2, 1, 1, bias=False)
        self.dbn2 = nn.BatchNorm2d(num_filters)
        #self.unpool2 = nn.ConvTranspose2d(num_filters, num_filters, 2, 2, bias=False)
        #self.unpoolbn2 = nn.BatchNorm2d(num_filters)

        self.dc3 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        #self.dc3 = nn.ConvTranspose2d(num_filters, num_filters, 3, 2, 1, 1, bias=False)
        self.dbn3 = nn.BatchNorm2d(num_filters)
        #self.unpool3 = nn.ConvTranspose2d(num_filters, num_filters, 2, 2, bias=False)
        #self.unpoolbn3 = nn.BatchNorm2d(num_filters)

        self.dc4 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        #self.dc4 = nn.ConvTranspose2d(num_filters, num_filters, 3, 2, 1, 1, bias=False)
        self.dbn4 = nn.BatchNorm2d(num_filters)
        #self.unpool4 = nn.ConvTranspose2d(num_filters, num_filters, 2, 2, bias=False)
        #self.unpoolbn4 = nn.BatchNorm2d(num_filters)

        self.dc5 = nn.Conv2d(num_filters, 1, 3, padding=1)
        #self.dc5 = nn.ConvTranspose2d(num_filters, 1, 3, 2, 1, 1, bias=False)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.pool(self.relu(self.bn1(self.c1(x))))
        h2 = self.pool(self.relu(self.bn2(self.c2(h1))))
        h3 = self.pool(self.relu(self.bn3(self.c3(h2))))
        h4 = self.pool(self.relu(self.bn4(self.c4(h3))))
        h4_flat = h4.view(h4.size(0), -1)
        mu = self.l_mu(h4_flat)
        log_var = self.l_log_var(h4_flat)
        return mu, log_var

    def reparametrize(self, mu, log_var):
        std = (log_var / 2.0).exp_()
        if self.use_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    # def decode(self, z):
    #     h4_flat = self.relu(self.dbn1(self.dl1(z)))
    #     h4 = h4_flat.view(h4_flat.size(0), self.num_filters, 2, 2)
    #     h4_up = self.relu(self.unpoolbn1(self.unpool1(h4)))
    #
    #     h3 = self.relu(self.dbn2(self.dc2(h4_up)))
    #     h3_up = self.relu(self.unpoolbn2(self.unpool2(h3)))
    #
    #     h2 = self.relu(self.dbn3(self.dc3(h3_up)))
    #     h2_up = self.relu(self.unpoolbn3(self.unpool3(h2)))
    #
    #     h1 = self.relu(self.dbn4(self.dc4(h2_up)))
    #     h1_up = self.relu(self.unpoolbn4(self.unpool4(h1)))
    #
    #     x = self.sigmoid(self.dc5(h1_up)[:, :, 2:-2, 2:-2])
    #     return x

    def decode(self, z):
        h4_flat = self.relu(self.dbn1(self.dl1(z)))
        h4 = h4_flat.view(h4_flat.size(0), self.num_filters, 2, 2)
        h4_up = self.unpool(h4)
        h3 = self.unpool(self.relu(self.dbn2(self.dc2(h4_up))))
        h2 = self.unpool(self.relu(self.dbn3(self.dc3(h3))))
        h1 = self.unpool(self.relu(self.dbn4(self.dc4(h2))))
        x = self.sigmoid(self.dc5(h1)[:, :, 2:-2, 2:-2])
        return x

    # def decode(self, z):
    #     h4_flat = self.relu(self.dbn1(self.dl1(z)))
    #     h4 = h4_flat.view(h4_flat.size(0), self.num_filters, 2, 2)
    #     h3 = self.relu(self.dbn2(self.dc2(h4)))
    #     h2 = self.relu(self.dbn3(self.dc3(h3)))
    #     h1 = self.relu(self.dbn4(self.dc4(h2)))
    #     x = self.sigmoid(self.dc5(h1)[:, :, 2:-2, 2:-2])
    #     return x

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparametrize(mu, log_var)
        return self.decode(z), mu, log_var
