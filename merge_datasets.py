import warnings
import bisect
from typing import List, Iterable, Dict, Tuple, Set

from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm


def _get_classes_from_dataset(dataset: Dataset) -> List[int]:
    all_labels: Set = set()
    for idx in tqdm(range(len(dataset)), "accumulating classes from subset"):
        _, y = dataset[idx]
        all_labels.add(y)
    sorted_labels = list(all_labels)
    sorted_labels.sort()
    assert len(sorted_labels) == len(set(sorted_labels))
    return sorted_labels


def obtain_class_mapping(ds: List[Dataset]) -> Dict[int, Dict[int, int]]:
    class_idx = 0
    mapping: Dict[int, Dict[int, int]] = {i: {} for i in range(len(ds))}
    for idx, d in enumerate(tqdm(ds, "reading dataset classes")):
        labels = _get_classes_from_dataset(d)
        for y in labels:
            if y in mapping[idx]:
                raise ValueError("Duplicated class found")
            mapping[idx][y] = class_idx
            class_idx += 1
    return mapping


class MergeDataset(Dataset):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """
    datasets: List[Dataset]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[Dataset], class_mapping: Dict[int, Dict[int, int]]) -> None:
        super(MergeDataset, self).__init__()
        self.datasets = list(datasets)
        self.class_mapping = class_mapping
        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore[arg-type]
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        X, y = self.datasets[dataset_idx][sample_idx]
        y_true = self.class_mapping[dataset_idx][y]
        return X, y_true

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes