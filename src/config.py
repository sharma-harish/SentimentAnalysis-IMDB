import transformers

MAX_LEN = 512
TRAIN_BATCH_SIZE = 1
VALID_BATCH_SIZE = 1
EPOCHS = 2
# ACCUMULATION = 2
BERT_PATH = 'F:\\Workspace\\SentimentAnalysis\\input\\bert-base-uncased'
MODEL_PATH = 'model.bin'
TRAINING_FILE = 'F:\\Workspace\\SentimentAnalysis\\input\\IMDB Dataset.csv'
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case=True
)