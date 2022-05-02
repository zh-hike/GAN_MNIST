import torch
from dataset.dataLoader import DL
from network import G, D, ConvD, ConvG
import torch.nn as nn
import torch.nn.functional as F
from vision import plot_img
import random
from torchvision.utils import save_image


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cpu')
        if args.device != -1:
            self.device = torch.device('cuda:%d' % self.args.device)
        self._init_data()
        self._init_model()

    def _init_data(self):
        data = DL(self.args)
        self.dl = data.dl
        self.n_classes = data.n_classes
        self.out_features = data.out_features

    def _init_model(self):
        # self.gen = G(in_features=self.args.dim, out_features=self.out_features)
        self.gen = ConvG(self.args.dim + self.n_classes, 1)
        self.gen.to(self.device)
        # self.dis = D(in_features=self.out_features)
        self.dis = ConvD()
        self.dis.to(self.device)
        self.g_opt = torch.optim.Adam(self.gen.parameters(), lr=self.args.glr, betas=(0.5, 0.999))
        self.d_opt = torch.optim.Adam(self.dis.parameters(), lr=self.args.dlr, betas=(0.5, 0.999))
        self.cri = nn.BCELoss()

    def val(self, epoch):
        imgs = torch.empty((0, 1, 28, 28))
        for time in range(5):
            mask = torch.eye(self.n_classes, dtype=torch.long).to(self.device).view(self.n_classes, self.n_classes, 1, 1)
            z = torch.randn((self.n_classes, self.args.dim, 1, 1)).to(self.device)
            z = torch.cat([mask, z], dim=1)
            output = self.gen(z)
            imgs = torch.cat([imgs, output.detach().cpu()], dim=0)

        # imgs = imgs.view(-1, 1, 28, 28)
        # plot_img(imgs, self.args.dataset, epoch)
        save_image(imgs, 'results/val/%d_print.png' % epoch, nrow=self.n_classes, normalize=True)

    def save_model(self):
        torch.save(self.gen.state_dict(), 'results/gen.pt')
        torch.save(self.dis.state_dict(), 'results/dis.pt')

    def train(self):
        patten = 'Iter: %d/%d   [==============]  D_loss: %.4f  G_loss: %.4f   D(x): %.5f    D(G(x)): %.5f'
        for epoch in range(self.args.epochs):
            g_scores = []
            d_scores = []
            D_loss = 0
            G_loss = 0
            for batch, (inputs, targets) in enumerate(self.dl):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                b = inputs.shape[0]
                # inputs = inputs.view(b, -1)
                z = torch.randn((b, self.args.dim, 1, 1)).to(self.device)

                mask = torch.zeros((b, self.n_classes, 1, 1), dtype=torch.float32).to(self.device)
                mask[list(range(b)), targets.squeeze()] = 1.
                # mask[mask == 1] = 0.7
                # mask[mask == 0] = 0.3 / (self.n_classes - 1)
                z = torch.cat([mask, z], dim=1)
                x_fake = self.gen(z)


                is_exchange = False

                # update D
                real_s = self.dis(inputs).squeeze()
                real = 1 - torch.rand_like(real_s).to(self.device) / 10
                fake = torch.zeros_like(real_s)
                if random.random() < self.args.label_exchange:
                    is_exchange = True
                    real, fake = fake, real
                d_real_loss = self.cri(real_s, real)
                fake_s = self.dis(x_fake.detach()).squeeze()
                d_fake_loss = self.cri(fake_s, fake)
                d_loss = d_real_loss + d_fake_loss
                D_loss += d_loss.item()
                self.d_opt.zero_grad()
                d_loss.backward()
                self.d_opt.step()
                d_scores.append(real_s.mean().detach().cpu())

                if is_exchange:
                    real, fake = fake, real
                # update G
                z = torch.randn((b, self.args.dim, 1, 1)).to(self.device)
                z = torch.cat([mask, z], dim=1)
                x_fake = self.gen(z)
                fake_s = self.dis(x_fake).squeeze()
                g_fake_loss = self.cri(fake_s, real)
                self.g_opt.zero_grad()
                g_fake_loss.backward()
                G_loss += g_fake_loss.item()
                self.g_opt.step()

                g_scores.append(fake_s.mean().detach().cpu())
                if batch == 0:
                    save_image(inputs, 'results/val/%d_real.png'%(epoch), nrow=20, normalize=True)
                    save_image(x_fake, 'results/val/%d_fake.png' % (epoch), nrow=20, normalize=True)
            n = len(g_scores)
            print(patten % (
                epoch,
                self.args.epochs,
                D_loss,
                G_loss,
                sum(d_scores) / n,
                sum(g_scores) / n,
            ))

            self.val(epoch)
            if epoch % 20 == 0:
                self.save_model()
