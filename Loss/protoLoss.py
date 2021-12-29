import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def Proto_Loss(num_way, num_of_support, num_of_query, support_features, query_features):

    n_support = num_of_support
    n_query = num_of_query
    n_class = num_way
    z_dim = support_features.size(-1)

    target_inds = torch.arange(0, n_class).view(
        n_class, 1, 1).expand(n_class, n_query, 1).long()
    target_inds = Variable(target_inds, requires_grad=False).cuda()
    proto = support_features.view(n_class, n_support, z_dim).mean(1)
    dists = euclidean_dist(query_features, proto)
    log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

    _, y_hat = log_p_y.max(2)
    acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()
    return loss_val, acc_val

# class Proto_Loss(nn.Module):
#     def __init__(self, num_way=5, num_of_support=5, num_of_query=10):
#         super().__init__()
#         self.N_Way = num_way
#         self.N_Support = num_of_support
#         self.N_Query = num_of_query

#     def forward(self, support_features, support_target, query_features,
#                 query_target):
#         kbits = support_features.size(1)

#         target_inds = torch.arange(0,
#                                    self.N_Way).view(self.N_Way, 1, 1).expand(
#                                        self.N_Way, self.N_Query, 1).long()
#         target_inds = Variable(target_inds, requires_grad=False).cuda()
#         print("target_inds {}".format(target_inds.size()))
#         support_features = support_features.view(self.N_Way, self.N_Support,
#                                                  kbits)
#         print("support_features {}".format(support_features.size()))
#         support_proto = support_features.mean(1)
#         print("support_proto {}".format(support_proto.size()))
#         dists = euclidean_dist(query_features, support_proto)
#         print("dists {}".format(dists.size()))
#         log_p_y = F.log_softmax(-dists, dim=1).view(self.N_Way, self.N_Query,
#                                                     -1)
#         loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
#         _, y_hat = log_p_y.max(2)
#         acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()
#         return loss_val, acc_val
