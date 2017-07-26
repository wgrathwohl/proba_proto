"""
Hierarchical VAE for training on Omniglot and MiniImagenet
"""
from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import os
import numpy as np
from networks import OmniglotModel
import utils
import data.omniglot as omniglot
from tensorboard_logger import configure, log_value

parser = argparse.ArgumentParser(description='test vae')
parser.add_argument('--train-shot', type=int, default=5, metavar='TRS',
                    help='number of samples per class per training batch (default: 5)')
parser.add_argument('--train-way', type=int, default=60, metavar='TRW',
                    help='number of classes to sample in a training batch (default: 60)')
parser.add_argument('--test-shot', type=int, default=5, metavar='TES',
                    help='number of samples per class per test batch (default: 5)')
parser.add_argument('--test-way', type=int, default=5, metavar='TEW',
                    help='number of classes to sample in a testing batch (default: 5)')
parser.add_argument('--dataset', type=str, default="omniglot", metavar='DS',
                    help='dataset to train on (default: Omniglot)')
parser.add_argument('--train-dir', type=str, default="/tmp/hvae_train", metavar='TD',
                    help='where to save logs and checkpointed models')
parser.add_argument('--epochs', type=int, default=25, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='LI',
                    help='how many batches to wait before logging training status')
parser.add_argument('--test-interval', type=int, default=2000, metavar='TEI',
                    help='how many batches to wait before running testing')
parser.add_argument('--test-episodes', type=int, default=1000, metavar='TEE',
                    help='how many episodes to average over when reporting testing accuracy')
parser.add_argument('--learn-class-var', type=bool, default=True, metavar='LCV',
                    help='whether or not to train the class variance')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    print("Using CUDA")
    torch.cuda.manual_seed(args.seed)


if args.dataset == "omniglot":
    train_dataset = omniglot.OmniglotDataset(args.train_way, "train", args.train_way * args.train_shot)
    test_dataset = omniglot.OmniglotDataset(args.test_way, "test", args.test_way * (args.test_shot + 1))
else:
    assert False

model = OmniglotModel(use_cuda=args.cuda)
# create Variable for log_class_var
# initial estimate for class std dev is .1
init_log_class_var = np.array([np.log(.1**2)], dtype=np.float32)

if args.cuda:
    log_class_var = Variable(
        torch.FloatTensor(init_log_class_var).cuda(),
        requires_grad=args.learn_class_var
    )
    model.cuda()
else:
    log_class_var = Variable(
        torch.FloatTensor(init_log_class_var),
        requires_grad=args.learn_class_var
    )


reconstruction_function = nn.BCELoss()
reconstruction_function.size_average = False


def exp_lr_scheduler(opt, epoch, init_lr=0.001, lr_decay_epoch=2):
    """Decay learning rate by a factor of 0.5 every lr_decay_epoch epochs."""
    lr = init_lr * (0.5**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in opt.param_groups:
        param_group['lr'] = lr

    return opt


def loss_function(recon_x, x, mu, log_var, log_class_var):
    w, n = mu.size()[:2]
    BCE = reconstruction_function(recon_x, x) / (w * n)
    # E_q[log(p(z))] = logC - E_q[hat(z**2)]/(2 * class_var / n) + ...
    # ... + E_q[hat(z)**2]/(2 * class_var / n) - E_q[hat(z)**2]/(2 * (1 + class_var / n))
    p1_array = np.array([-((.5 * (n - 1)) * np.log(2.0 * np.pi))], dtype=np.float32)
    if args.cuda:
        p1 = Variable(torch.FloatTensor(p1_array).cuda())
    else:
        p1 = Variable(torch.FloatTensor(p1_array))
    p2 = log_class_var.sub(np.log(n)).mul(.5)
    p3 = log_class_var.mul(-.5 * n)
    p4 = torch.log1p(log_class_var.exp().div(n)).add(np.log(2.0 * np.pi)).mul(-.5)
    print(type(p1.data), type(p2.data), type(p3.data), type(p4.data))
    logC = p1.add(p2).add(p3).add(p4)

    mu_sqr_plus_var = mu.mul(mu).sum(dim=1).add(log_var.exp().sum(dim=1))[:, 0, :]
    s = None
    for i in range(n-1):
        cur = mu[:, i, :]
        rest = mu[:, i+1:, :].sum(dim=1)[:, 0, :]
        this_term = cur.mul(rest)
        if i == 0:
            s = this_term
        else:
            s.add_(this_term)
    mu_term = s.mul(2.0)
    E_q_z_sqr_hat = mu_sqr_plus_var.div(n)
    E_q_z_hat_sqr = mu_sqr_plus_var.add(mu_term).div(n*n)

    # E_q[log(p(z))] = logC + t1 + t2 + t3
    denom12 = log_class_var.exp().div(n).mul(2.0).expand(E_q_z_sqr_hat.size())
    denom3 = log_class_var.exp().div(n).add(1.0).mul(2.0).expand(E_q_z_hat_sqr.size())
    t1 = E_q_z_sqr_hat.div(denom12).mul(-1.0)
    t2 = E_q_z_hat_sqr.div(denom12)
    t3 = E_q_z_hat_sqr.div(denom3).mul(-1.0)
    E_q_log_p_z = logC.expand(t1.size()).add(t1).add(t2).add(t3).sum()
    E_q_log_q_z = (log_var.add(1.0).add(np.log(2.0 * np.pi))).mul(-.5).sum()

    KLD = E_q_log_q_z.sub(E_q_log_p_z) / (w * n)

    return BCE + KLD, BCE, KLD

def accuracy_function(mu_tr, logvar_tr, mu_te, logvar_te):
    pass


def accuracy_function(**kwrgs):
    return 0.0


def train(epoch, opt):
    opt = exp_lr_scheduler(opt, epoch - 1, init_lr=1e-3)
    model.train()
    train_loss = 0
    for batch_idx in xrange(args.test_interval):
        data, _ = train_dataset.get_batch_points()
        data = Variable(data)
        # data should be (way, shot, c, h, w)
        # reshape to (way * shot, c, h, w) for encoding and decoding
        data_batch = data.view(
            data.size(0) * data.size(1),
            data.size(2), data.size(3), data.size(4)
        )
        if args.cuda:
            data = data.cuda()
            data_batch = data_batch.cuda()
        opt.zero_grad()
        recon_batch, mu, logvar = model(data_batch)
        # reshape mu and logvar back to (way, show, encoding_dim)
        mu = mu.view(args.train_way, args.train_shot, -1)
        logvar = logvar.view(args.train_way, args.train_shot, -1)
        loss, bce, kld = loss_function(recon_batch, data_batch, mu, logvar, log_class_var)
        loss.backward()
        train_loss += loss.data[0]
        opt.step()
        if batch_idx % args.log_interval == 0:
            step = (args.test_interval * (epoch - 1)) + batch_idx
            log_value("total_train_loss", loss.data[0], step)
            log_value("train_bce", bce.data[0], step)
            log_value("train_kld", kld.data[0], step)
            log_value("log_class_var", log_class_var.data[0], step)
            print("Logging to step {}".format(step))
            c, h, w = recon_batch.size()[1:]
            utils.save_recons_few_shot(
                data.data, recon_batch.view(args.train_way, args.train_shot, c, h, w).data,
                os.path.join(args.train_dir, "epoch_{}_iter_{}".format(epoch, batch_idx))
            )
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tBCE: {:.6f}, KLD: {:6f}'.format(
                epoch, batch_idx * len(data), len(train_dataset),
                100. * batch_idx / args.test_interval,
                loss.data[0], bce.data[0], kld.data[0]))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / args.test_interval))

def log_expected_prob(mu1, logvar1, mu2, logvar2):
    """

    :param mu1:
    :param logvar1:
    :param mu2:
    :param logvar2:
    :return: log(Ex_N(mu1, logvar1)[N(mu2, logvar2)]) the log of the expected probability under normal distribution 1
    of a sample drawn from normal distribution 2
    """
    logvar2_expanded = logvar2.expand(logvar1.size())
    mu2_expanded = mu2.expand(mu1.size())

    sum_vars = logvar1.exp().add(logvar2_expanded.exp())
    const = torch.log(sum_vars.mul(2.0 * np.pi)).mul(-.5)
    mu_diffs = mu1.sub(mu2_expanded)
    exponent = mu_diffs.mul(mu_diffs).div(sum_vars.mul(2.0)).mul(-1.0)
    total_per_dim = const.add(exponent)
    total = total_per_dim.sum(dim=2)[:, :, 0]
    SM = torch.nn.Softmax()
    sm = SM(total.view(1, -1)).view(total.size())
    smm = sm.sum(dim=1)[:, 0]
    return smm


def test(epoch):
    model.eval()
    accuracies = []
    for i in xrange(args.test_episodes):
        data, _ = test_dataset.get_batch_points()
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        data_batch = data.view(
            data.size(0) * data.size(1),
            data.size(2), data.size(3), data.size(4)
        )
        recon_batch, mu, logvar = model(data_batch)
        mu = mu.view(data.size(0), data.size(1), -1)
        logvar = logvar.view(data.size(0), data.size(1), -1)
        mu_tr, mu_te = mu[:, :-1, :], mu[:, -1, :]
        logvar_tr, logvar_te = logvar[:, :-1, :], logvar[:, -1, :]
        for j in range(mu_te.size(0)):
            mu_j_te, logvar_j_te = mu_te[j, :], logvar_te[j, :]
            test_j = log_expected_prob(mu_tr, logvar_tr, mu_j_te.view(1, 1, -1), logvar_j_te.view(1, 1, -1))
            pred_ind = np.argmax(test_j.data.numpy())
            if pred_ind == j:
                accuracies.append(1.0)
            else:
                accuracies.append(0.0)

    test_accuracy = np.mean(np.array(accuracies))

    log_value("test_accuracy", test_accuracy, epoch * args.test_interval)
    print('====> Test set accuracy: {:.4f}'.format(test_accuracy))

if __name__ == "__main__":
    if os.path.exists(args.train_dir):
        print("Deleting existing train dir")
        import shutil
        shutil.rmtree(args.train_dir)

    os.makedirs(args.train_dir)
    configure(args.train_dir, flush_secs=5)
    if args.learn_class_var:
        optimizer = optim.Adam(list(model.parameters()) + [log_class_var])
    else:
        optimizer = optim.Adam(model.parameters())
    for epoch in range(1, args.epochs + 1):
        train(epoch, optimizer)
        test(epoch)
