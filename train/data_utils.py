import torch
from collections import defaultdict


def individual_collate(batch):
    """
    Custom collation function for collate with new implementation of individual samples in data pipeline
    """

    data = batch

    collected_data = defaultdict(list)

    for i in range(len(list(data))):
        for k in data[i].keys():
            collected_data[k].append(data[i][k])

    for k in collected_data.keys():
        collected_data[k] = torch.stack(collected_data[k])

    return collected_data
