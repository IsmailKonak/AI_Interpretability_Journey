import torch as t
from torch.utils.data import Dataset



class MedianDataset(Dataset):

    def __init__(self, size: int, max_num: int, length: int, seed: int = 42):
        '''
        Dataset of the form [a1, ..., an, median], where the median
        of the numbers with digits a equals the median value.

        The digits in a are uniformly chosen from 0 to max_num inclusive.
        '''
        self.vocab = [str(i) for i in range(max_num)]
        self.size = size
        self.max_num = max_num
        if length < 3:
            raise ValueError("Length must be at least 3")
        if length % 2 != 1:
            raise ValueError("Length must be odd")
        self.length = length
        t.manual_seed(seed)  # for reproducible results

        # Generate random digits for 'a'
        numbers = t.randint(low=0, high=max_num, size=(size, length))

        # Calculate the median of the numbers
        median_of_nums = numbers.median(dim=1).values.unsqueeze(1)

        # Concatenate the numbers and their median
        self.toks = t.concat([numbers, median_of_nums], dim=1)

        # Convert the result back to tokens
        self.str_toks = [
            [self.vocab[tok] for tok in toks]
            for toks in self.toks
        ]

    def __getitem__(self, index):
        return self.toks[index]

    def __len__(self):
        return self.size

    def to(self, device: str):
        self.toks = self.toks.to(device)
        return self



