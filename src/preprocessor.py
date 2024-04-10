import pandas as pd
from transformers import T5Tokenizer


class Preprocessor:
    def __init__(self, tokenizer_model='t5-small', max_len=512):
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_model)
        self.max_len = max_len

    def preprocess(self, data):
        """
        Preprocesses the data for the AbstractiveModel.
        :param data: A pandas DataFrame with columns 'Article' and 'Summary'.
        :return: A DataFrame with tokenized and formatted data for training.
        """
        # Create empty lists to hold tokenized data
        tokenized_articles = []
        tokenized_summaries = []

        for _, row in data.iterrows():
            content_text = row['Content']  # Update from 'Article' to 'Content'
            summary_text = row['Summary']

            # Tokenize and format the article and summary text
            article_enc = self.tokenizer.encode_plus(
                content_text,
                max_length=self.max_len,
                pad_to_max_length=True,
                return_attention_mask=True,
                truncation=True
            )

            summary_enc = self.tokenizer.encode_plus(
                summary_text,
                max_length=self.max_len,
                pad_to_max_length=True,
                return_attention_mask=True,
                truncation=True
            )

            tokenized_articles.append({
                'input_ids': article_enc['input_ids'],
                'attention_mask': article_enc['attention_mask']
            })

            tokenized_summaries.append({
                'input_ids': summary_enc['input_ids'],
                'attention_mask': summary_enc['attention_mask']
            })

        # Convert lists to DataFrame for easier handling
        preprocessed_data = pd.DataFrame({
            'article': tokenized_articles,
            'summary': tokenized_summaries
        })

        return preprocessed_data
