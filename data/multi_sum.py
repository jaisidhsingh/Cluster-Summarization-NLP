from datasets import load_dataset


class MultiSumDataset():
	def __init__(self, dataset_name="", split="train"):
		self.dataset_name = dataset_name
		self.split = split
		
		self.dataset = load_dataset(self.dataset_name, split=self.split)
		self.inputs = self.dataset['document']
		self.targets = self.dataset['summary']

	def __getitem__(self, idx):
		return self.inputs[idx], self.targets[idx]