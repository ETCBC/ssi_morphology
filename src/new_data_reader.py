"""Read and prepare new data on which a model can make predictions."""

class CustomDataReader:
    def __init__(self, input_filename, seq_len):

        self.input_filename = input_filename
        self.seq_len = seq_len
        self.input_verses = self.import_data()
        self.data_ids2text = {}
        self.data_ids2labels = {}
        self.prepared_data = {}

        self.read_data()
        self.make_sequences()

    def import_data(self):
        with open(self.input_filename, 'r') as f:
            return f.readlines()

    def read_data(self):
        word_id = 0
        for i in range(len(self.input_verses)):
            bo, ch, ve, text = tuple(self.input_verses[i].strip().split('\t'))
            split_text = text.split()

            for word in split_text:
                self.data_ids2text[word_id] = word
                self.data_ids2labels[word_id] = [bo, ch, ve]
                word_id += 1

    def make_sequences(self):
        word_list = []
        idx_list = []
        for idx, text in self.data_ids2text.items():
            word_list.append(text)
            idx_list.append(idx)
            if len(word_list) == self.seq_len:
                self.prepared_data[tuple(idx_list)] = ' '.join(word_list)
                word_list = []
                idx_list = []
        if idx_list:
            self.prepared_data[tuple(idx_list)] = ' '.join(word_list)
            
class HebrewWordsNewText(Dataset):
    """A Pytorch wrapper around the hebrew bible text. Processed per word."""

    def __init__(self, data: dict,
                  INPUT_WORD_TO_IDX: dict, OUTPUT_WORD_TO_IDX: dict):
        """
        Args:
            input_filename (str)
        Note: output is reduced using the mc_reduce function
        The dataset contains hashes with:
            text: str
            output: str
            encoded_text: Tensor
            encoded_output: Tensor
        """

        self.word_indices = list(data.keys())
        self.word_texts = list(data.values())
        self.INPUT_WORD_TO_IDX = INPUT_WORD_TO_IDX
        self.OUTPUT_WORD_TO_IDX = OUTPUT_WORD_TO_IDX

    def __len__(self):
        return len(self.word_indices)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        text = self.word_texts[idx]
        sample = {
                "text": text,
                "encoded_text": encode_string(text, self.INPUT_WORD_TO_IDX, add_sos=False, add_eos=True),
                "indices": self.word_indices[idx]
                }

        return sample