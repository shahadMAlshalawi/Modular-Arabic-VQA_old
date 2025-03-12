from .okvqa_dataset import OKVQADataset

class VQAv2Dataset(OKVQADataset):
    #TODO: GPT Captioning
    def __init__(self, BDS, VDS, GPT4oDS):
        #TODO: GPT Captioning
        super().__init__(BDS=BDS,VDS=VDS,GPT4oDS=GPT4oDS)