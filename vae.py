"""
Run a convolutional-vae on mnist for testing purposes
"""
from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import os
from networks import OmniglotModel
import utils
import data.omniglot as omniglot
from tensorboard_logger import configure, log_value

parser = argparse.ArgumentParser(description='test vae')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--dataset', type=str, default="omniglot", metavar='DS',
                    help='dataset to train on (default: Omniglot)')
parser.add_argument('--train-dir', type=str, default="/tmp/vae_train_learned_upsample", metavar='TD',
                    help='where to save logs and checkpointed models')
parser.add_argument('--epochs', type=int, default=25, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


if args.dataset == "mnist":
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/mnist', train=True, download=False,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/mnist', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
elif args.dataset == "omniglot":
    train_loader = omniglot.OmniglotUnsupervisedLoader("train", args.batch_size)
    test_loader = omniglot.OmniglotUnsupervisedLoader("test", args.batch_size)
else:
    assert False

model = OmniglotModel(cuda=args.cuda)
if args.cuda:
    model.cuda()

reconstruction_function = nn.BCELoss()
reconstruction_function.size_average = False


def exp_lr_scheduler(opt, epoch, init_lr=0.001, lr_decay_epoch=5):
    """Decay learning rate by a factor of 0.5 every lr_decay_epoch epochs."""
    lr = init_lr * (0.5**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in opt.param_groups:
        param_group['lr'] = lr

    return opt


def loss_function(recon_x, x, mu, log_var):
    BCE = reconstruction_function(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return BCE + KLD

optimizer = optim.Adam(model.parameters())

def train(epoch, opt):
    opt = exp_lr_scheduler(opt, epoch - 1, init_lr=1e-3)
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        opt.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        opt.step()
        if batch_idx % args.log_interval == 0:
            step = (len(train_loader) * (epoch - 1)) + batch_idx
            log_value("train_loss", loss.data[0] / len(data), step)
            print("Logging to step {}".format(step))
            utils.save_recons(
                data.data, recon_batch.data,
                os.path.join(args.train_dir, "epoch_{}_iter_{}".format(epoch, batch_idx))
            )
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    for data, _ in test_loader:
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).data[0]

    test_loss /= len(test_loader.dataset)
    log_value("test_loss", test_loss, epoch-1)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    if os.path.exists(args.train_dir):
        print("Deleting existing train dir")
        import shutil
        shutil.rmtree(args.train_dir)

    os.makedirs(args.train_dir)
    configure(args.train_dir, flush_secs=5)

    for epoch in range(1, args.epochs + 1):
        train(epoch, optimizer)
        test(epoch)
