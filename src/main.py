from data_loader import DataLoader
from preprocessor import Preprocessor
from evaluator import Evaluator
from models import ExtractiveModel
from models import AbstractiveModel
from models import HybridModel


def main():
    # Initialize components
    loader = DataLoader()
    preprocessor = Preprocessor()
    evaluator = Evaluator()

    # Load and preprocess data
    train_data, test_data = loader.load_data("data/train_data.txt")
    train_data = preprocessor.preprocess(train_data)
    test_data = preprocessor.preprocess(test_data)

    # Initialize models
    models = {
        "Extractive": ExtractiveModel(),
        "Abstractive": AbstractiveModel(),
        "Hybrid": HybridModel()
    }

    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"Training {model_name} Model")
        model.train(train_data)

        print(f"Evaluating {model_name} Model")
        score = model.evaluate(test_data)
        print(f"{model_name} Model Score: {score}")


if __name__ == "__main__":
    main()
