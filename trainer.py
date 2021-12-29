import torch
import torch.nn as nn
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from models.Network import MetaNet, ClsNet
from models.resnet18 import Resnet18
from dataset.CustomData import MyCustomDataset
from dataset.MetaData import MetaSamples
from Loss.meta_loss import Meta_Loss
from Loss.quantization import MSE_Quantization, Bitwise_Quantization
from utils.LoadWeights import load_preweights
from progress.bar import Bar
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
torch.cuda.manual_seed_all(1)


def accuracy(logits, target):
    predict = torch.argmax(logits.data, 1)
    correct = (predict == target.data).sum().item() / target.size(0)
    return correct


def relaxtion(real_hash, beta):
    relaxed = torch.tanh(beta * real_hash)
    return relaxed


def train(args):
    # prepare some base configurations, includes gpu, and parameters direction
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    parameters_dir = os.path.join(
        args.root, args.parameters, args.dataset, str(args.K_bits) + "kbits")
    if os.path.exists(parameters_dir) is False:
        os.makedirs(parameters_dir)
    transform = transforms.Compose([
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    # loading the training set
    trainset = MyCustomDataset(root_path=args.img_tr, transform=transform)

    # loading the testing set
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    testset = MyCustomDataset(root_path=args.img_te,
                              transform=test_transform)

    # loading the networks
    hash_model = Resnet18(K_bits=args.K_bits).cuda().float()
    model_dict = hash_model.state_dict()
    pretrained_dict = torch.load(args.pretrained)
    state_dict = {}
    for key, value in model_dict.items():
        if key in pretrained_dict and value.size(
        ) == pretrained_dict[key].size():
            state_dict[key] = pretrained_dict[key]
            print("loading weights {}, size {}".format(
                key, pretrained_dict[key].size()))
        else:
            state_dict[key] = model_dict[key]

    hash_model.load_state_dict(state_dict)

    cls_model = ClsNet(K_bits=args.K_bits,
                       num_classes=args.label_dim).cuda().float()
    optimizer = torch.optim.Adam([{'params': hash_model.parameters()}, {'params': cls_model.parameters()}],
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[5000], gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    BESTACC = 0
    iteration = 0
    while iteration <= args.Iteration:
        # beta = args.beta * (10 ** (iteration // 5000))
        beta = args.beta

        lr = optimizer.state_dict()['param_groups'][0]['lr']

        N_way = args.N_way
        categories = [i for i in range(args.label_dim)]
        random.shuffle(categories)
        num_split = args.label_dim // N_way
        subset_catgories = []
        for i in range(num_split):
            start = i * N_way
            end = (i + 1) * N_way
            if end >= args.label_dim:
                pass
            subset_catgories.append(categories[start:end])

        if args.dataset == "AID_05":
            N_support = 3
            N_query = 2
        if args.dataset == "AID_08":
            N_support = 4
            N_query = 4
        if args.dataset == "AID_10":
            N_support = 5
            N_query = 5

        MeatTrain = MetaSamples(trainset,
                                num_way=N_way,
                                num_of_support=N_support,
                                num_of_query=N_query)

        # trainloader's length is one
        accumulated_accuracy = 0
        for k, subset in enumerate(subset_catgories):
            hash_model.train()
            cls_model.train()
            (SupportImg, SupportTarget), (QueryImg,
                                          QueryTarget) = MeatTrain.__getitem__(subset)
            SupportImg = SupportImg.squeeze().cuda()
            QueryImg = QueryImg.squeeze().cuda()

            SupportTarget = SupportTarget.squeeze()
            QueryTarget = QueryTarget.squeeze()

            QueryRealHash = hash_model(QueryImg)
            QueryHash = relaxtion(QueryRealHash, beta=beta)
            QueryLogits = cls_model(QueryHash)

            SupportRealHash = hash_model(SupportImg)
            SupportHash = relaxtion(SupportRealHash, beta=beta)
            SupportLogits = cls_model(SupportHash)

            cellloss, IntraMetaDist, QueryMetaDist, DifMetaDist = Meta_Loss(
                num_way=N_way, num_of_support=N_support, num_of_query=N_query, CategorySet=subset, margin=args.margin, support_features=SupportHash, support_target=SupportTarget, query_features=QueryHash, query_target=QueryTarget)

            SupportTarget = SupportTarget.cuda()
            QueryTarget = QueryTarget.cuda()
            SupportCEloss = criterion(SupportLogits, SupportTarget)
            QueryCEloss = criterion(QueryLogits, QueryTarget)
            ce_loss = SupportCEloss + QueryCEloss
            acc = (accuracy(SupportLogits, SupportTarget) +
                   accuracy(QueryLogits, QueryTarget)) / 2.

            loss = cellloss + ce_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            iteration += 1

            accumulated_accuracy += acc
            average_accumulated_accuracy = accumulated_accuracy / (k + 1)

            with open(os.path.join(parameters_dir, 'train_log.csv'),
                      'a') as f:
                log = 'Meta-Hashing::: | iter:{} | Iteration:{} | beta:{} |lr:{} | Total-Loss:{:.4f} | Meta-Loss:{:.4f} | Meta-inter:{:.4f} | Meta-intra:{:.4f} | CE-loss:{:.4f} | Class-acc:{:.4f}'.format(
                    iteration, args.Iteration, beta, lr, loss.item(), cellloss.item(), IntraMetaDist.item() + QueryMetaDist.item(), DifMetaDist.item(), ce_loss.item(), average_accumulated_accuracy)
                f.write(log + '\n')
            information = 'Meta-Hashing: ({iteration}/{size})|beta:{beta}|lr:{lr}|Total-Loss:{tloss:.4f}|Meta-Loss:{mloss:.4f}|Meta-inter:{inter:.4f}|Meta-intra:{intra:.4f}|CE-loss:{closs:.4f}|Class-acc:{cls:.4f}'.format(
                iteration=iteration,
                size=args.Iteration,
                beta=beta,
                lr=lr,
                tloss=loss.item(),
                mloss=cellloss.item(),
                inter=IntraMetaDist.item() + QueryMetaDist.item(),
                intra=DifMetaDist.item(),
                closs=ce_loss.item(),
                cls=average_accumulated_accuracy)
            print(information)

            if (iteration >= (args.Iteration - 1000) and iteration % 10 == 0) or iteration % 500 == 0:
                testloader = DataLoader(testset,
                                        batch_size=128,
                                        shuffle=True,
                                        num_workers=args.num_workers)
                hash_model.eval()
                cls_model.eval()
                bar = Bar('Testing stage {}/{}'.format(iteration, args.Iteration),
                          max=testloader.__len__())
                testset_total_correct = 0
                for i, (data, target) in enumerate(testloader):
                    data, target = data.cuda(), target.cuda()
                    RealHash = hash_model(data)
                    HashCodes = relaxtion(RealHash, beta=beta)
                    Logits = cls_model(HashCodes)
                    acc = accuracy(Logits, target)
                    testset_total_correct += acc
                    average_testset_total_correct = testset_total_correct / \
                        (i + 1)
                    with open(os.path.join(parameters_dir, 'test_log.csv'),
                              'a') as f:
                        log = 'Testing::: | Epoch:{} | Batch:{} | Class-acc:{:.4f}'.format(
                            iteration, i, average_testset_total_correct)
                        f.write(log + '\n')
                    bar.suffix = 'Testing: ({batch}/{size})|Class-acc:{cls:.4f}'.format(
                        batch=i + 1,
                        size=testloader.__len__(),
                        cls=average_testset_total_correct)
                    bar.next()
                bar.finish()

                if BESTACC < average_testset_total_correct:
                    BESTACC = average_testset_total_correct
                    print(
                        "This is new upgraded model {}, and the parameters are saved!!"
                        .format(BESTACC))
                    # save the optimized model
                    torch.save(hash_model.state_dict(),
                               os.path.join(parameters_dir, "hash_model.pth"))
                    torch.save(cls_model.state_dict(),
                               os.path.join(parameters_dir, "cls_model.pth"))
                    # save the optimized model
                torch.save(hash_model.state_dict(),
                           os.path.join(parameters_dir, "hash_model_last.pth"))
                torch.save(cls_model.state_dict(),
                           os.path.join(parameters_dir, "cls_model_last.pth"))
