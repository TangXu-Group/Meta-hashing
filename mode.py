import argparse
import os
from trainer import train
from extractcode import extract_code

root = "/home/admin1/PytorchProject/Meta-Hashing/Meta-hash-AID/"
pretrained = "/home/admin1/PytorchProject/data/pytorch_weights/resnet18-5c106cde.pth"
parser = argparse.ArgumentParser(description='Meta-Hashing-pretrained')
parser.add_argument(
    '--phase',
    default=0,
    type=int,
    help="0 means training, 1 means extract hash codes"
)
# define some basic configurations
parser.add_argument('--gpus', default='1', type=str)
parser.add_argument('--root', default=root, type=str)
parser.add_argument('--data_dir', default=root + '/dataset/', type=str)
parser.add_argument('--pretrained', default=pretrained, type=str)
parser.add_argument('--dataset', default='NWPU', type=str)
parser.add_argument('--parameters', default='parameters', type=str)
# define the codes dir
parser.add_argument('--codes_dir', default='codes', type=str)

# define the basic-parameters
parser.add_argument('--Iteration', default=10000, type=int)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--weight_decay', default=0.0005, type=float)
parser.add_argument('--num_workers', default=4, type=int)

# define the hyper-parameters for training
parser.add_argument('--K_bits', default=24, type=int)
parser.add_argument('--N_way', default=5, type=int)
parser.add_argument('--alpha', default=1.0, type=float)
parser.add_argument('--margin', default=24.0, type=float)
parser.add_argument('--beta', default=1.0, type=float)

# define the hyper-parameters for extracting the hash codes
parser.add_argument('--batchsize', default=100, type=int)

args = parser.parse_args()
label_dict = {'UC_Merced_05': 21, 'UC_Merced_08': 21, 'UC_Merced_10': 21, 'NWPU_05': 45,
              'NWPU_08': 45, 'NWPU_10': 45, 'AID_05': 30, 'AID_08': 30, 'AID_10': 30}
args.label_dim = label_dict[args.dataset]
args.img_tr = os.path.join(args.data_dir, args.dataset, "train.txt")
args.img_te = os.path.join(args.data_dir, args.dataset, "test.txt")


for item in vars(args):
    print("item {}, value {}".format(item, vars(args)[item]))

if args.phase == 0:
    train(args)
elif args.phase == 1:
    extract_code(args)
