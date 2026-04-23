import os
import pickle
import torch
from torch.utils.data import Dataset
from am_baseline.problem.state import StateTSP


class TSP(object):
    NAME = 'tsp'

    @staticmethod
    def get_costs(dataset, pi):
        # Validate tour: must contain exactly nodes 0..n-1
        assert (
            torch.arange(pi.size(1), device=pi.device).view(1, -1).expand_as(pi) ==
            pi.data.sort(1)[0]
        ).all(), "Invalid tour"

        # Gather coordinates in tour order
        d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))

        # Tour length = sum of consecutive distances + return to start
        return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1), None

    @staticmethod
    def get_edge_costs(dataset, pi):
        """
        Per-edge costs along the tour, for Stage 1 value-head targets.

        Returns a (batch, N) tensor where entry [b, t] is the cost of the edge
        traversed at decoding step t+1:
            edges[:, 0 .. N-2] = consecutive distances between visited nodes
            edges[:, N-1]      = closing edge from last visited node back to first

        Sanity: get_edge_costs(...).sum(dim=1) == get_costs(...)[0] within fp.
        """
        d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))
        forward_edges = (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2)          # (B, N-1)
        closing_edge = (d[:, 0] - d[:, -1]).norm(p=2, dim=1).unsqueeze(-1)  # (B, 1)
        return torch.cat([forward_edges, closing_edge], dim=1)           # (B, N)

    @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTSP.initialize(*args, **kwargs)


class TSPDataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(TSPDataset, self).__init__()

        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset + num_samples])]
        else:
            self.data = [torch.FloatTensor(size, 2).uniform_(0, 1) for i in range(num_samples)]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
