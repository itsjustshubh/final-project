from abc import ABC, abstractmethod


class SummarizationModel(ABC):

    def __init__(self, model_name):
        self.model_name = model_name

    @abstractmethod
    def train(self, train_data):
        pass

    @abstractmethod
    def generate_summary(self, input_text):
        pass

    @abstractmethod
    def save_model(self, file_path):
        pass

    @abstractmethod
    def load_model(self, file_path):
        pass

    @abstractmethod
    def evaluate(self, test_data):
        pass
