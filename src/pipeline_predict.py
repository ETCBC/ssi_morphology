import os

from config import device, PREDICTION_DATA_FOLDER
from evaluate_transformer import translate
from new_data_reader import ConfigParser, ModelImporter, NewDataReader, HebrewWordsNewText

class PipeLinePredict:
    def __init__(self, predict_config_file):
        self.predict_config_file = predict_config_file
    
        self.config_parser = self.parse_config()
        self.model_importer = self.load_model()
        
        self.new_data_reader = self.read_new_data()
        self.new_dataset = self.make_new_dataset()
    
        self.make_predictions_transformer_model()
    
    def parse_config(self):
        config_parser = ConfigParser(self.predict_config_file)
        return config_parser
        
    def load_model(self):
        model_importer = ModelImporter(self.config_parser.model_config_data,
                                       self.config_parser.model_folder, 
                                       self.config_parser.model_name)
        return model_importer
        
    def read_new_data(self):
        new_data_reader = NewDataReader(self.config_parser.new_data_file, 
                                        self.config_parser.model_config_data['seq_len'])
        return new_data_reader
                                        
    def make_new_dataset(self):
        hebrew_words_new_text = HebrewWordsNewText(self.new_data_reader.prepared_data,
                                                   self.new_data_reader.data_ids2labels,
                                                   self.config_parser.model_config_data['input_w2idx'],
                                                   self.config_parser.model_config_data['output_w2idx'])
        return hebrew_words_new_text
        
    def make_predictions_transformer_model(self):
        model = self.model_importer.loaded_transformer
        model.eval()
        
        new_data_file = self.config_parser.new_data_file.split('.')[0]

        with open(f'{PREDICTION_DATA_FOLDER}/results_predictions_{new_data_file}.txt', 'w') as f:
            for i in range(len(self.new_dataset)):
                predicted = translate(model.to(device), self.new_dataset[i]['encoded_text'].to(device),
                                  self.new_dataset.OUTPUT_IDX_TO_WORD, 
                                  self.new_dataset.OUTPUT_WORD_TO_IDX)
                indices = self.new_dataset[i]['indices']
                labels = self.new_dataset[i]['labels']
                input_text = self.new_dataset[i]['text']
                
                predicted_separate_words = predicted.split()
                input_text_separate_words = input_text.split()
                if len(predicted_separate_words) == len(input_text_separate_words):
                    for idx, label, input_txt, pred_txt in zip(indices, labels, input_text_separate_words, predicted_separate_words):
                        f.write(f'{str(idx)}\t{" ".join(label)}\t{input_txt}\t{pred_txt}\n')
                else:
                    f.write(f'{str(indices)}\t{str(labels)}\t{text}\{predicted}\n')
             