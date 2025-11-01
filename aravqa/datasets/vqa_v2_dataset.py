from .okvqa_dataset import OKVQADataset

class VQAv2Dataset(OKVQADataset):
    def __init__(self, BDS, VDS, GPT4oDS):
        super().__init__(BDS=BDS,VDS=VDS,GPT4oDS=GPT4oDS)