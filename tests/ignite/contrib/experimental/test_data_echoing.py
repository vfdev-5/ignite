import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from ignite.contrib.experimental import ExampleEchoingSampler, MemoizingDataset


class DummyDataset(Dataset):

    def __init__(self, size, with_index=False):
        self.data = torch.rand(size, 3, 12, 12)
        self.targets = torch.randint(0, 10, size=(size, ))

    def __getitem__(self, i):
        return self.data[i], self.targets[i]

    def __len__(self):
        return len(self.data)


def test_example_echoing_sampler_no_base_sampler():

    torch.manual_seed(12)

    num_echoes = 4
    size = 100
    dataset = DummyDataset(size=size)
    sampler = ExampleEchoingSampler(num_echoes=num_echoes, dataset_length=len(dataset))
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=16, num_workers=4)

    for batch in dataloader:
        batch_x, batch_y = batch
        batch_xy_hash = [(hash(x.numpy().tobytes()), y) for x, y in zip(batch_x, batch_y)]
        unique_batch_xy_hash = list(set(batch_xy_hash))

        assert len(unique_batch_xy_hash) * num_echoes >= len(batch_xy_hash)
        for xy_hash in unique_batch_xy_hash:
            count = batch_xy_hash.count(xy_hash)
            assert count >= num_echoes, "{} : {}".format(xy_hash, batch_xy_hash)


def test_example_echoing_sampler_with_base_sampler():

    num_echoes = 4
    size = 100
    dataset = DummyDataset(size=size)

    torch.manual_seed(12)
    weights = torch.arange(0, size)
    base_sampler = WeightedRandomSampler(weights, num_samples=size)
    original_dataloader = DataLoader(dataset, sampler=base_sampler, batch_size=16 // num_echoes, num_workers=4)

    seen_batches_hash = []
    for batch in original_dataloader:
        batch_x, batch_y = batch
        batch_xy_hash = [(hash(x.numpy().tobytes()), y) for x, y in zip(batch_x, batch_y)]
        seen_batches_hash.append(batch_xy_hash)

    torch.manual_seed(12)
    sampler = ExampleEchoingSampler(num_echoes=num_echoes, base_sampler=base_sampler)
    echoing_dataloader = DataLoader(dataset, sampler=sampler, batch_size=16, num_workers=4)

    for seen_batch, batch in zip(seen_batches_hash, echoing_dataloader):
        batch_x, batch_y = batch
        batch_xy_hash = [(hash(x.numpy().tobytes()), y) for x, y in zip(batch_x, batch_y)]

        for xy_hash in seen_batch:
            count = batch_xy_hash.count(xy_hash)
            assert count >= num_echoes, "{} : {}".format(xy_hash, batch_xy_hash)
