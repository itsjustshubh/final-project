# abstractive_model.py
from model import SummarizationModel
from data_loader import CSVDataLoader
from custom_dataset import CustomDataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd


class AbstractiveModel(SummarizationModel):
    def __init__(self, tokenizer_model='t5-small', preprocessor=None):
        super().__init__("AbstractiveModel")
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_model)
        self.model = T5ForConditionalGeneration.from_pretrained(
            tokenizer_model)
        self.preprocessor = preprocessor

    def train(self, train_data_path, epochs=1, batch_size=4, learning_rate=5e-5, nrows=None):
        data_loader = CSVDataLoader(
            preprocessor=self.preprocessor)  # Updated class name
        train_data = data_loader.load_data(train_data_path, nrows=nrows)

        train_dataset = CustomDataset(train_data)
        train_loader = DataLoader(train_dataset, batch_size=batch_size)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.model.train()

        for epoch in range(epochs):
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch['article_input_ids'].to(torch.device("cpu"))
                attention_mask = batch['article_attention_mask'].to(
                    torch.device("cpu"))
                labels = batch['summary_input_ids'].to(torch.device("cpu"))

                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    def generate_summary(self, input_text):
        input_ids = self.tokenizer.encode(
            "summarize: " + input_text, return_tensors="pt", add_special_tokens=True)
        summary_ids = self.model.generate(input_ids)
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def save_model(self, file_path):
        self.model.save_pretrained(file_path)

    def load_model(self, file_path):
        self.model = T5ForConditionalGeneration.from_pretrained(file_path)
        self.tokenizer = T5Tokenizer.from_pretrained(file_path)

    def evaluate(self, test_data):
        # Basic placeholder for the evaluate method
        print("Evaluation logic is not yet implemented.")
        # Implement the evaluation logic using a suitable metric (e.g., ROUGE)
        return 0  # Example return value


# Example usage
if __name__ == "__main__":
    from preprocessor import Preprocessor

    preprocessor = Preprocessor()
    model = AbstractiveModel(preprocessor=preprocessor)

    nrows_for_training = 10  # Specify the number of rows to train on
    model.train("data/data.csv", epochs=3,
                batch_size=2, nrows=nrows_for_training)

    large_text = "In a significant development today, scientists at the International Institute of Advanced Research announced a groundbreaking discovery in the field of renewable energy. The new technology, which harnesses the power of ocean currents, promises to revolutionize the energy industry. Dr. Emily Stanton, the lead researcher on the project, explained that the technology works by utilizing underwater turbines to generate electricity. These turbines are designed to be highly efficient and environmentally friendly, offering a sustainable alternative to fossil fuels. In conclusion, while there are hurdles to overcome, the promise of harnessing the power of ocean currents for renewable energy is a beacon of hope. This innovative approach could play a crucial role in the global transition to sustainable energy, marking a significant step forward in the effort to combat climate change and protect our planet for future generations."
    summary = model.generate_summary(
        large_text)
    print(f"Generated summary: {summary}")
