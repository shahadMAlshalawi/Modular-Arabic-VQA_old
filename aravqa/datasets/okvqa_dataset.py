import torch

class OKVQADataset(torch.utils.data.Dataset):
    #TODO: GPT Captioning
    def __init__(self, BDS, VDS, GPT4oDS):
        self.BDS = BDS
        self.VDS = VDS
        self.GPT4oDS = GPT4oDS
       

    def __len__(self):
        return len(self.BDS)

    def __getitem__(self, idx):
        example = {
            "metadata": self.BDS[idx]["metadata"],
            "image": self.BDS[idx]["image"],
            "question": self.BDS[idx]["question"],
            "answers": self.BDS[idx].get("answers",[]),
            "bit": self.BDS[idx].get("captions",[]),
            "violet": self.VDS[idx].get("captions",[]),
            "GPT4o": self.GPT4oDS[idx].get("captions",[])
        }
        return example