import torch
import numpy as np
import random, json
from tqdm import tqdm
from transformers import DistilBertTokenizer
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6, flush=True)
    os.remove('temp.p')

def get_device():
    """
    Returns the appropriate device for the environment.
    Most likely, the device will be CUDA in training and CPU in production.
    """
    if torch.cuda.is_available():        
        device = torch.device("cuda")
        print('Use GPU:', torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print('No GPU available, using CPU instead.')
    return device

def set_seed(seed=None):
    """
    Set all seeds to make results reproducible.
    When SEED is None, does not set seed.

    Params:
    - int seed: seed for pseudorandom generator
    """
    if seed:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

TOKENIZER_CLASSES = {
  'distilbert-base-cased': DistilBertTokenizer,
}

def get_dataloader(data_file, model_type, batch_size=32, num_points=None, val_size=None):
    """
    Returns a dataloader for the data in DATA_FILE.

    Params:
    - str data_file: path to data file (assumed to be json for this project)
    - str model_type: the type of model that the dataloader will be used for
    - int batch_size: number of points per batch
    - int num_points: number of total points to include in the dataloader (if None, all points are used)
    - float val_size: proportion of points to use for validation (if None, no validation dataloader is created)
    """
    print(f"Creating dataloader for {data_file} ...")

    # TODO: Define test-time behavior (no ratings)

    # Collect the review texts and ratings
    reviews = []
    ratings = []
    n = 0
    with open(data_file, "r") as f:
        for line in tqdm(f):
            review = json.loads(line)
            reviews.append(review['text'])
            ratings.append(review['stars'])
            n += 1
            # Only collect NUM_POINTS reviews
            if num_points and n >= num_points:
                break
    
    # Tokenize the review texts
    # TODO: Look at https://huggingface.co/transformers/training.html for a cleaner way
    # to tokenize + get attention mask
    tokenizer = TOKENIZER_CLASSES[model_type].from_pretrained(model_type)
    encoded_reviews = []
    for review in tqdm(reviews):
        encoding = tokenizer.encode(review, padding='max_length')   # TODO: figure out more efficient padding scheme
        MAX_LEN = TOKENIZER_CLASSES[model_type].max_model_input_sizes[model_type]
        encoding = encoding[:MAX_LEN]   # truncate encoding to max length allowed by model
        encoded_reviews.append(encoding)

    # Construct attention masks to allow model to distinguish data from padding
    attention_masks = []
    for review in encoded_reviews:   
        mask = [int(token_id != tokenizer.pad_token_id) for token_id in review]
        attention_masks.append(mask)
    
    # Create dataloader(s)
    if val_size:
        NUM_TRAIN_POINTS = int( (1.0 - val_size) * len(reviews) )
        train_data = TensorDataset(
            torch.tensor(encoded_reviews[:NUM_TRAIN_POINTS]),
            torch.tensor(attention_masks[:NUM_TRAIN_POINTS]),
            torch.tensor(ratings_to_labels(ratings[:NUM_TRAIN_POINTS]))
        )
        val_data = TensorDataset(
            torch.tensor(encoded_reviews[NUM_TRAIN_POINTS:]),
            torch.tensor(attention_masks[NUM_TRAIN_POINTS:]),
            torch.tensor(ratings_to_labels(ratings[NUM_TRAIN_POINTS:]))
        )
        return (
            DataLoader(train_data, sampler=SequentialSampler(train_data), batch_size=batch_size),
            DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=batch_size)
        )
    else:
        data = TensorDataset(
            torch.tensor(encoded_reviews),
            torch.tensor(attention_masks),
            torch.tensor(ratings_to_labels(ratings))
        )
        return DataLoader(data, sampler=SequentialSampler(data), batch_size=batch_size)

RATINGS = [
    1, 2, 3, 4, 5
]

def ratings_to_labels(ratings):
    """
    Converts a list of ratings into labels (indices).
    """
    return [RATINGS.index(rating) for rating in ratings]


def labels_to_ratings(labels):
    """
    Converts a list of labels (indices of RATINGS)
    into the corresponding ratings.
    """
    return np.take(RATINGS, labels)


def accuracy(logits, labels):
  logits = logits.detach().cpu().numpy()
  preds = np.argmax(logits, axis=1)
  labels = labels.detach().cpu().numpy()
  acc = np.mean(preds == labels)
  return acc
