import random

from torch.utils.data.sampler import Sampler
import torch.utils.data as tud


class BatchRandomSampler(Sampler):
    """
    Batches the data consecutively and randomly samples
    by batch without replacement.
    """

    def __init__(self, data_source, batch_size):
        super(BatchRandomSampler, self).__init__(data_source)
        it_end = len(data_source) - batch_size + 1
        self.batches = [range(i, i + batch_size) for i in range(0, it_end, batch_size)]
        self.data_source = data_source

    def __iter__(self):
        random.shuffle(self.batches)
        return (i for b in self.batches for i in b)

    def __len__(self):
        return len(self.data_source)


class SortedDataset(tud.Dataset):

    def __init__(self, list_of_examples, lengths_list, bucket_diff=4):

        max_len = max(lengths_list)

        num_buckets = max_len // bucket_diff
        buckets = [[] for _ in range(num_buckets)]
        for example in list_of_examples:
            bid = min(len(example) // bucket_diff, num_buckets - 1)
            buckets[bid].append(example)

        # Sort by input length followed by output length
        sort_fn = lambda x: len(x)
        for b in buckets:
            b.sort(key=sort_fn)
        data = [d for b in buckets for d in b]
        self.data = data

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        # todo ICULOS always continous?
        inputs = self.data[idx].drop(['SepsisLabel', 'ICULOS'], axis=1).values
        targets = self.data[idx]['SepsisLabel'].values

        return inputs, targets


def make_raw_loader(list_of_examples, lengths_list, is_sepsis,
                    batch_size,
                    num_workers=4,
                    use_sampler=True):

    dataset = SortedDataset(list_of_examples=list_of_examples, lengths_list=lengths_list)

    if use_sampler:
        sampler = BatchRandomSampler(dataset, batch_size)
    else:
        sampler = None

    loader = tud.DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
                            collate_fn=lambda batch: zip(*batch), drop_last=True)

    return loader
