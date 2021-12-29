import torch
import torch.nn as nn


def Meta_Loss(num_way, num_of_support, num_of_query, CategorySet, margin, support_features, support_target, query_features, query_target):
    SupportIndsList = list(
        map(
            lambda c: list(
                support_target.eq(c).nonzero().squeeze().numpy()),
            CategorySet))
    QueryIndsList = list(
        map(lambda c: list(query_target.eq(c).nonzero().squeeze().numpy()),
            CategorySet))

    IntraMetaDist, QueryMetaDist = calculate_same_dist(
        support_features, support_target, SupportIndsList, query_features, query_target, QueryIndsList)
    DifMetaDist = calculate_different_dist(
        support_features, support_target, SupportIndsList, query_features, query_target, QueryIndsList, margin)

    TotalDist = IntraMetaDist + QueryMetaDist + DifMetaDist
    return TotalDist, IntraMetaDist, QueryMetaDist, DifMetaDist


def calculate_same_dist(support_features, support_target, SupportIndsList,
                        query_features, query_target, QueryIndsList):
    IntraMetaDist, QueryMetaDist = [], []
    assert len(SupportIndsList) == len(QueryIndsList)
    for i, (SupportInds,
            QueryInds) in enumerate(zip(SupportIndsList, QueryIndsList)):
        #
        query_category = query_target[QueryInds]
        support_category = support_target[SupportInds]
        assert support_category.unique()[0] == query_category.unique()[0]
        MetaCell = support_features[SupportInds]
        MetaQuery = query_features[QueryInds]
        Dist = euclidean_dist(MetaQuery, MetaCell)
        max_indices = torch.argmax(Dist, dim=1, keepdim=False)
        min_indices = torch.argmin(Dist, dim=1, keepdim=False)
        max_samples = MetaCell[max_indices].squeeze()
        min_samples = MetaCell[min_indices].squeeze()
        mid_samples = (max_samples + min_samples) / 2.
        IntraMetaDist_temp = euclidean_dist_vector(
            max_samples, mid_samples) + euclidean_dist_vector(min_samples, mid_samples)
        QueryMetaDist_temp = euclidean_dist_vector(MetaQuery, mid_samples)
        IntraMetaDist.append(IntraMetaDist_temp)
        QueryMetaDist.append(QueryMetaDist_temp)
    IntraMetaDist = torch.stack(IntraMetaDist)
    QueryMetaDist = torch.stack(QueryMetaDist)

    return torch.mean(IntraMetaDist), torch.mean(QueryMetaDist)


def calculate_different_dist(support_features, support_target,
                             SupportIndsList, query_features, query_target,
                             QueryIndsList, margin):
    dist_hard = []
    for QueryInds in QueryIndsList:
        for SupportInds in SupportIndsList:
            query_category = query_target[QueryInds]
            support_category = support_target[SupportInds]

            if query_category.unique()[0] == support_category.unique()[0]:
                pass
            if query_category.unique()[0] != support_category.unique()[0]:
                MetaCell = support_features[SupportInds]
                MetaQuery = query_features[QueryInds]
                dist = euclidean_dist(MetaQuery, MetaCell)
                dist_temp, _ = torch.min(dist, dim=1)
                # dist_temp = torch.min(dist)
                dist_hard.append(dist_temp)
    dist_hard = torch.stack(dist_hard)
    dist_hard = torch.relu(margin - dist_hard)
    return torch.mean(dist_hard)


def euclidean_dist(x, y, squared=True):
    """
    Compute (Squared) Euclidean distance between two tensors.

    Args:
        x: input tensor with size N x D.
        y: input tensor with size M x D.

        return: distance matrix with size N x M.
    """
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception('Invalid input shape.')

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    dist = torch.pow(x - y, 2).sum(2)

    if squared:
        return dist
    else:
        return torch.sqrt(dist + 1e-12)


def euclidean_dist_vector(x, y, squared=True):
    """
    Compute (Squared) Euclidean distance between two tensors.

    Args:
        x: input tensor with size N x D.
        y: input tensor with size N x D.

        return: distance matrix with size N.

    For example:
        distance(x1, y1)
        distance(x2, y2)
        distance(x3, y3)
    """
    n1, d1 = x.size(0), x.size(1)
    n2, d2 = y.size(0), y.size(1)
    if n1 != n2 and d1 != d2:
        raise Exception('Invalid input shape.')

    dist = torch.pow(x - y, 2).sum(1)

    if squared:
        return dist
    else:
        return torch.sqrt(dist + 1e-12)
