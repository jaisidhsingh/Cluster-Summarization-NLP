from datasets import load_dataset


class PubMedDataset():
	def __init__(self, dataset_name="ccdv/pubmed-summarization", split="train"):
		self.dataset_name = dataset_name
		self.split = split
		self.dataset = load_dataset(self.dataset_name, 
			split=self.split
		)

		self.inputs = self.dataset['article']
		self.targets = self.dataset['abstract']
	
	def __getitem__(self, idx):
		return self.inputs[idx].replace("\n", " "), self.targets[idx].replace("\n", " ")

	def __len__(self):
		return len(self.dataset)
