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

def clean_text(origin_text):
    # Remove punctuation and illegal characters
    text = re.sub("[^a-zA-Z]", " ", origin_text)
    # Convert all characters to lowercase, and perform word segmentation through space characters
    words = text.lower().split()
    # remove stopwords
    stop_words = set(stopwords.words("english"))
    meaningful_words = [w for w in words if w not in stop_words]
    # back to str
    cleaned_text = " ".join(meaningful_words)
    return cleaned_text


def split_json(infile, train_file, val_file, ratio=0.8):
    print("split the train data to train and val")
    train_df = pd.read_json(infile, encoding='utf-8')
    train_df['is_spoiler'] = train_df['is_spoiler'].apply(lambda x: 1 if x else 0)
    train_df['review_text'] = train_df['review_text'].apply(lambda x: clean_text(x))
    idxs = np.arange(train_df.shape[0])
    np.random.shuffle(idxs)
    val_size = int(len(idxs) * ratio)
    train = train_df.iloc[idxs[:val_size], :]
    train[['review_text', 'is_spoiler']].to_csv(train_file, index=False)
    val = train_df.iloc[idxs[val_size:], :]
    val[['review_text', 'is_spoiler']].to_csv(val_file, index=False)

def clean_test(infile, outfile):
    print("clean the test data")
    test_df = pd.read_json(infile, encoding='utf-8')
    test_df['review_text'] = test_df['review_text'].apply(lambda x: clean_text(x))
    test_df[['review_text']].to_csv(outfile, index=False)

def tokenizer(text):  # create a tokenizer function
        #  a list of <class 'spacy.tokens.token.Token'>
        return [tok.text for tok in spacy_en.tokenizer(text) ]

#load the data iteratively
def load_data(args, logger):
    logger.info('load the data iteratively')
    review_text = data.Field(sequential=True, lower=True, tokenize=tokenizer, fix_length=args.max_review_len)
    is_spoiler = data.LabelField(dtype=torch.int)
    review_text.tokenize = tokenizer
    logger.info('use the tabularDataset')
    train, val, test = data.TabularDataset.splits(
        path='./data/',
        skip_header=True,
        train='dataset_train.csv',
        validation='dataset_val.csv',
        test='dataset_test.csv',
        format='csv',
        fields=[('review_text', review_text), ('is_spoiler', is_spoiler)])

    if args.static:
        logger.info('load the word vector from the pretrained')
        review_text.build_vocab(train, val,
                                vectors=Vectors(name='glove.6B.100d.txt', cache='./pretrain/'),
                                unk_init=torch.Tensor.normal_)
        args.embedding_dim = review_text.vocab.vectors.size()[-1]
        args.vectors = review_text.vocab.vectors
        logger.info("使用预训练词库")
    else:
        review_text.build_vocab(train, val)
        logger.info('自行构建词库')

    is_spoiler.build_vocab(train)
    # iterator
    train_iter, val_iter, test_iter = data.Iterator.splits(
        (train, val), sort_key=lambda x: len(x.review_text),
        batch_sizes=(args.train_batch_size, args.val_batch_size, args.test_batch_size), device=0)

    args.vocab_size = len(review_text.vocab)
    logger.info('the size of vocab is {}'.format(len(review_text.vocab)))

    return train_iter, val_iter, test_iter

# split_json(infile='../data/train.json',
#           train_file='../data/dataset_train.csv',
#           val_file='../data/dataset_val.csv',
#           ratio=0.8)
# clean_test(infile='../data/test.json',
#            outfile='../data/dataset_test.csv',)

def load_train_data(logger, infile):
    logger.info('load the train and val data')
    train_df = pd.read_json(infile, encoding='utf-8')
    train_df['is_spoiler'] = train_df['is_spoiler'].apply(lambda x: 1 if x else 0)
    train_df['review_text'] = train_df['review_text'].apply(lambda x: clean_text(x))
    train_data = train_df[['review_text', 'is_spoiler']].values.tolist()
    return train_data
