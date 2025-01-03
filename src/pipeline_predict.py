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
    
        self.export_predictions_transformer_model()
    
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
        
    def export_predictions_transformer_model(self):
        """
        Export data that are created by make_predictions function.
        Results are written to an output file if this is specified in YAML file,
        else results are written to standard output.
        """
        new_data_file = self.config_parser.new_data_file.split('.')[0]
        
        if self.config_parser.output:
            with open(f'{PREDICTION_DATA_FOLDER}/{self.config_parser.output}', 'w') as f:
                for prediction in self.make_predictions():
                    f.write(prediction + '\n')
        else:
            for prediction in self.make_predictions():
                print(prediction)
    
    def make_predictions(self):
        """
        Generator function which makes predictions on new data using trained model.
        """
        model = self.model_importer.loaded_transformer
        model.eval()
        
        predict_idx = self.config_parser.predict_idx

        for i in range(len(self.new_dataset)):
            predicted = translate(model.to(device), 
                                  self.new_dataset[i]['encoded_text'].to(device),
                                  self.new_dataset.OUTPUT_IDX_TO_WORD,
                                  self.new_dataset.OUTPUT_WORD_TO_IDX,
                                  self.config_parser.beam_size,
                                  self.config_parser.beam_alpha,
                                  self.config_parser.language,
                                  self.config_parser.version
                                  )
            indices = self.new_dataset[i]['indices']
            labels = self.new_dataset[i]['labels']
            input_text = self.new_dataset[i]['text']
                
            predicted_separate_words = predicted.split(' ')
            input_text_separate_words = input_text.split(' ')
            
            if predict_idx + 1 <= len(predicted_separate_words) and predict_idx + 1 <= len(input_text_separate_words):
                if i == 0:
                    for idx in range(predict_idx+1):
                        yield f'{str(indices[idx])}\t{" ".join(labels[idx])}\t{input_text_separate_words[idx]}\t{predicted_separate_words[idx]}'
                elif i == len(self.new_dataset) - 1:
                    for idx in range(predict_idx, len(labels)):
                        yield f'{str(indices[idx])}\t{" ".join(labels[idx])}\t{input_text_separate_words[idx]}\t{predicted_separate_words[idx]}'
                else:
                    yield f'{str(indices[predict_idx])}\t{" ".join(labels[predict_idx])}\t{input_text_separate_words[predict_idx]}\t{predicted_separate_words[predict_idx]}'
            else:
                yield f'{str(indices)}\t{str(labels)}\t{input_text}\t{predicted}'
