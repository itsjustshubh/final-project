from models.model import SummarizationModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pickle


class ExtractiveModel(SummarizationModel):
    def __init__(self):
        super().__init__("ExtractiveModel")
        self.vectorizer = TfidfVectorizer()
        self.classifier = RandomForestClassifier()

    def train(self, articles, summaries):
        sentences, labels = self.preprocess(articles, summaries)
        X = self.vectorizer.fit_transform(sentences)
        self.classifier.fit(X, labels)

    def generate_summary(self, input_text):
        sentences = input_text.split('. ')  # Simple sentence tokenizer.
        X = self.vectorizer.transform(sentences)
        predictions = self.classifier.predict(X)

        summary_sentences = [sentence for sentence, label in zip(
            sentences, predictions) if label == 1]
        return ' '.join(summary_sentences)

    def save_model(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump((self.vectorizer, self.classifier), file)

    def load_model(self, file_path):
        with open(file_path, 'rb') as file:
            self.vectorizer, self.classifier = pickle.load(file)

    def evaluate(self, test_articles, test_summaries):
        test_sentences, test_labels = self.preprocess(
            test_articles, test_summaries)
        X_test = self.vectorizer.transform(test_sentences)
        predictions = self.classifier.predict(X_test)
        return accuracy_score(test_labels, predictions)
