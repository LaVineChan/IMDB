import re
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import torch
import torchtext
from torchtext import data
import spacy
import en_core_web_sm
spacy_en = en_core_web_sm.load()
from torchtext.vocab import Vectors
from tqdm import tqdm

def clean_text(origin_text):
    # Remove punctuation and illegal characters
    text = re.sub("[^a-zA-Z]", " ", origin_text)
    # Convert all characters to lowercase, and perform word segmentation through space characters
    cleaned_text = text.lower().split()
    # remove stopwords
    # stop_words = set(stopwords.words("english"))
    # meaningful_words = [w for w in words if w not in stop_words]
    # # back to str
    # cleaned_text = " ".join(meaningful_words)
    return cleaned_text


def split_train_val(infile, logger, ratio=0.8):
    logger.info("split the train data to train and val")
    data_df = pd.read_json(infile, encoding='utf-8')
    data_df['is_spoiler'] = data_df['is_spoiler'].apply(lambda x: 1 if x else 0)
    data_df['review_text'] = data_df['review_text'].apply(lambda x: clean_text(x))
    idxs = np.arange(data_df.shape[0])
    np.random.shuffle(idxs)
    val_size = int(len(idxs) * ratio)
    train = data_df.iloc[idxs[:val_size], :]
    train_df = train[['review_text', 'is_spoiler']]
    val = data_df.iloc[idxs[val_size:], :]
    val_df = val[['review_text', 'is_spoiler']]
    return train_df, val_df

def get_test(test_file):
    print("clean the test data")
    test_df = pd.read_json(test_file, encoding='utf-8')
    test_df['review_text'] = test_df['review_text'].apply(lambda x: clean_text(x))
    test_df = test_df[['review_text']]
    return test_df

def tokenizer(text):
    # create a tokenizer function
    #  a list of <class 'spacy.tokens.token.Token'>
    return [tok.text for tok in spacy_en.tokenizer(text) ]


def get_dataset(csv_data, review_text, is_spoiler, test=False):
    fields = [('review_text', review_text), ('is_spoiler', is_spoiler)]
    examples = []

    if test:
        for text in tqdm(csv_data['review_text']):
            examples.append(data.Example.fromlist([text, None], fields))
    else:
        for text, label in tqdm(zip(csv_data['review_text'], csv_data['is_spoiler'])):
            examples.append(data.Example.fromlist([text, label], fields))
    return examples, fields

#load the data iteratively
def load_data(args, logger):
    logger.info('load the data iteratively')
    review_text = data.Field(sequential=True, lower=True, tokenize=tokenizer, fix_length=args.max_review_len)
    is_spoiler = data.Field(sequential=False, use_vocab=False, dtype=torch.int)
    train_file = './data/train.json'
    test_file = './data/test.json'
    train_df, val_df = split_train_val(train_file, logger, ratio=args.split_ratio)
    test_df = get_test(test_file)
    # 得到构建Dataset所需的examples和fields
    train_examples, train_fields = get_dataset(train_df, review_text, is_spoiler)
    valid_examples, valid_fields = get_dataset(val_df, review_text, is_spoiler)
    test_examples, test_fields = get_dataset(test_df, review_text, is_spoiler=None, test=True)

    # 构建Dataset数据集
    train = data.Dataset(train_examples, train_fields)
    val = data.Dataset(valid_examples, valid_fields)
    test = data.Dataset(test_examples, test_fields)

    if args.static:
        logger.info('load the word vector from the pretrained')
        review_text.build_vocab(train, val, test,
                                vectors=Vectors(name='glove.6B.100d.txt', cache='./pretrain/'),
                                unk_init=torch.Tensor.normal_)
        args.embedding_dim = review_text.vocab.vectors.size()[-1]
        args.vectors = review_text.vocab.vectors
        logger.info("使用预训练词库")
    else:
        review_text.build_vocab(train, val, test)
        logger.info('自行构建词库')

    is_spoiler.build_vocab(train)
    # iterator
    train_iter, val_iter = data.BucketIterator.splits(
        (train, val), sort_key=lambda x: len(x.review_text),
        batch_sizes=(args.train_batch_size, args.val_batch_size), device=0, sort_within_batch=False,)
    test_iter = data.Iterator(test, batch_size=args.test_batch_size, device=0, sort=False, sort_within_batch=False, repeat=False)
    args.vocab_size = len(review_text.vocab)
    logger.info('the size of vocab is {}'.format(len(review_text.vocab)))

    return train_iter, val_iter, test_iter


