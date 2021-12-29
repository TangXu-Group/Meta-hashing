import numpy as np
from torch.utils.data import Dataset, Subset, DataLoader
from random import sample
import torch


class MetaSamples(Dataset):
    def __init__(self, BasicDataset, num_way, num_of_support, num_of_query):
        """
        BasicDataset: the class method for MyCustomDataset
                    It is used for reading the training and testing sets from the files.txt

        num_way: int
            N-way. The number of sampled categories in each episode

        num_of_support: int
             The number of support samples for each category within N-way

        num_of_query: int
             The number of query samples for each category within N-way

        """
        self.N_way, self.S, self.Q = num_way, num_of_support, num_of_query
        self.BasicDataset = BasicDataset
        self.transform = self.BasicDataset.transform
        self.labels = self.BasicDataset.img_label
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {
            label: list(np.where(np.array(self.labels) == label)[0])
            for label in self.labels_set
        }
        if self.N_way > len(self.labels_set):
            raise ValueError(
                'N-way is larger than the total number of category')

    def __getitem__(self, categories):
        # random sample N_way categories from whole data sets
        S_ind, Q_ind = [], []

        for item in categories:
            ind = self.label_to_indices[item]

            if (self.S + self.Q) > len(ind):
                raise ValueError(
                    'the size of support and query is larger than total number within tarining set'
                )
            support_ind = sample(ind, self.S)
            remained = [i for i in ind if i not in support_ind]
            query_ind = sample(remained, self.Q)
            # append the global support index list and query index list
            S_ind += support_ind
            Q_ind += query_ind

        (SupportImg, SupportTarget) = self.__loader__(S_ind)
        (QueryImg, QueryTarget) = self.__loader__(Q_ind)
        return (SupportImg, SupportTarget), (QueryImg, QueryTarget)

    def __loader__(self, ind):
        # subset = torch.utils.data.Subset(self.BasicDataset, ind)
        # subset_loader = DataLoader(subset, subset.__len__())
        datas, targets = [], []
        for item in ind:
            data, target = self.BasicDataset.__getitem__(item)
            datas.append(data)
            targets.append(target)
        # for data, target in subset_loader:
        t_data = torch.stack(datas, dim=0)
        t_target = torch.stack(targets, dim=0)
        return t_data, t_target

    def __len__(self):
        return 1
