import torch
import numpy as np

from torch.utils.data.dataloader import default_collate


class Loader:
    def __init__(self, dataset, flags, device):
        self.device = device
        split_indices = list(range(len(dataset)))
        sampler = torch.utils.data.sampler.SubsetRandomSampler(split_indices)
        self.loader = torch.utils.data.DataLoader(dataset, batch_size=flags.batch_size, sampler=sampler,
                                             num_workers=flags.num_workers, pin_memory=flags.pin_memory,
                                             collate_fn=collate_events)

    def __iter__(self):
        for data in self.loader:
            data = [d.to(self.device) for d in data]
            yield data

    def __len__(self):
        return len(self.loader)


def collate_events(data):
    labels = []
    events1 = []
    events2 = []
    targets = []
    for i, d in enumerate(data):
        labels.append(d[3])
        targets.append(d[2])
        ev1 = np.concatenate([d[0], i*np.ones((len(d[0]),1), dtype=np.float32)],1)
        ev2 = np.concatenate([d[1], i*np.ones((len(d[1]),1), dtype=np.float32)],1)
        events1.append(ev1)
        events2.append(ev2)
    events1 = torch.from_numpy(np.concatenate(events1,0))
    events2 = torch.from_numpy(np.concatenate(events2,0))
    labels = default_collate(labels)
    targets = default_collate(targets)
    return events1, events2, targets, labels