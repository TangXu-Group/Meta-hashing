import torch
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from models.Network import MetaNet, ClsNet
from models.resnet18 import Resnet18
from dataset.CustomData import MyCustomDataset
from tqdm import tqdm
import numpy as np
import argparse
torch.cuda.manual_seed_all(1)


def relaxtion(real_hash, beta):
    relaxed = torch.tanh(beta * real_hash)
    return relaxed


def extract_code(args):
    # prepare some base configurations, includes gpu, and parameters direction
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # loading the training set
    trainset = MyCustomDataset(root_path=args.img_tr, transform=transform)
    trainloader = DataLoader(
        trainset, batch_size=args.batchsize, num_workers=4)

    # loading the testing set
    testset = MyCustomDataset(root_path=args.img_te, transform=transform)
    testloader = DataLoader(testset, batch_size=args.batchsize, num_workers=4)

    # loading the networks
    # hash_model = MetaNet(K_bits=args.K_bits).cuda().float()
    hash_model = Resnet18(K_bits=args.K_bits).cuda().float()
    parameters_dir = os.path.join(
        args.root, args.parameters, args.dataset, str(args.K_bits) + "kbits")
    hash_model.load_state_dict(
        torch.load(parameters_dir + "/hash_model.pth"))
    hash_model.eval()
    beta = args.beta

    trainfeatures = []
    with tqdm(trainloader, desc="testing stage") as iterator:
        for i, (data, target) in enumerate(iterator):
            data, target = data.cuda(), target.cuda()
            features = relaxtion(hash_model(data), beta=beta)
            features = features.cpu().detach().numpy()
            trainfeatures.extend(features)
    testfeatures = []
    with tqdm(testloader, desc="testing stage") as iterator:
        for i, (data, target) in enumerate(iterator):
            data, target = data.cuda(), target.cuda()
            features = relaxtion(hash_model(data), beta=beta)
            features = features.cpu().detach().numpy()
            testfeatures.extend(features)
    trainfeatures, testfeatures = np.array(trainfeatures), np.array(
        testfeatures)
    print("the training set features size {}, testing set features {}".format(
        trainfeatures.shape, testfeatures.shape))

    save_path = os.path.join(args.root, args.dataset, args.codes_dir)
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    np.save(save_path + "/traincodes.npy", trainfeatures)
    print("sucessfully generate trainfeatures")
    np.save(save_path + "/testcodes.npy", testfeatures)
    print("sucessfully generate test features")

    # hash_model = MetaNet(K_bits=args.K_bits).cuda().float()
    hash_model_last = Resnet18(K_bits=args.K_bits).cuda().float()
    parameters_dir = os.path.join(
        args.root, args.parameters, args.dataset, str(args.K_bits) + "kbits")
    hash_model_last.load_state_dict(
        torch.load(parameters_dir + "/hash_model_last.pth"))
    hash_model_last.eval()
    beta = args.beta

    trainfeatures_last = []
    with tqdm(trainloader, desc="testing stage") as iterator:
        for i, (data, target) in enumerate(iterator):
            data, target = data.cuda(), target.cuda()
            features = relaxtion(hash_model_last(data), beta=beta)
            features = features.cpu().detach().numpy()
            trainfeatures_last.extend(features)
    testfeatures_last = []
    with tqdm(testloader, desc="testing stage") as iterator:
        for i, (data, target) in enumerate(iterator):
            data, target = data.cuda(), target.cuda()
            features = relaxtion(hash_model_last(data), beta=beta)
            features = features.cpu().detach().numpy()
            testfeatures_last.extend(features)
    trainfeatures_last, testfeatures_last = np.array(trainfeatures_last), np.array(
        testfeatures_last)
    print("the training set features size {}, testing set features {}".format(
        trainfeatures_last.shape, testfeatures_last.shape))

    save_path = os.path.join(args.root, args.dataset, args.codes_dir)
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    np.save(save_path + "/traincodes_last.npy", trainfeatures_last)
    print("sucessfully generate trainfeatures")
    np.save(save_path + "/testcodes_last.npy", testfeatures_last)
    print("sucessfully generate test features")
