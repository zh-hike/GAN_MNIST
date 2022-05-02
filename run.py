import argparse
from Trainer import Trainer

parse = argparse.ArgumentParser('gan')
parse.add_argument('--epochs', type=int, help="训练epochs", default=500)
parse.add_argument('--glr', type=float, help="生成器学习率", default=1e-3)
parse.add_argument('--dlr', type=float, help="鉴别器学习率", default=1e-4)
parse.add_argument('--dim', type=int, help="随机生成的维度", default=128)
parse.add_argument('--device', type=int, default=0)
parse.add_argument('--label_exchange', type=float, help="标签反转的概率", default=0.05)
parse.add_argument('--data_path', type=str, help="数据集所在路径", default='../MyData/mnist/')
parse.add_argument('--dataset', type=str, choices=['mnist'], default='mnist')

args = parse.parse_args()

if __name__ == "__main__":
    trainer = Trainer(args)
    trainer.train()
