import random
import argparse
import logging
import time
from TextRNN.train import *
import numpy as np
from TextRNN.preprocess_new import load_data
from TextRNN.model import *


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()


parser = argparse.ArgumentParser(description='Tuning with LSTM')
parser.add_argument('--train', default="./data/train.json")
parser.add_argument('--test', default="./data/test.json")
parser.add_argument('--top_words', default=5000)
parser.add_argument('--max_review_len',default=500)
parser.add_argument('--split_ratio', default=0.8)
parser.add_argument('--train_batch_size', type=int, default=256)
parser.add_argument('--val_batch_size', type=int,  default=256)
parser.add_argument('--test_batch_size', type=int,  default=256)
parser.add_argument('--embedding_dim', type=int,  default=128)
parser.add_argument('--hidden_size', type=int,  default=64)
parser.add_argument('--layer_num', type=int,  default=2, help='the number of lstm layer')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--embedding-dim', type=int, default=128, help='embedding size of word vector')
parser.add_argument('--static', type=bool, default=True, help=' whether use the pretrained word vector')
parser.add_argument('--log_file', default='./log/', help=' the storage of log file')
parser.add_argument('--log_interval', type=int, default=200, help='经过多少iteration对训练集进行输出')
parser.add_argument('--test-interval', type=int, default=200, help='经过多少iteration对验证集进行测试')
parser.add_argument('--early-stop', type=int, default=5000, help='早停时迭代的次数')
parser.add_argument('--save_dir', default='./save/', help='模型存储位置')
parser.add_argument('--device', type=int, default=1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('--cuda', type=bool, default=True, help='是否使用cuda')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout')


args = parser.parse_args()


if not os.path.exists(args.log_file) :
	os.mkdir(args.log_file)

# 创建一个logger
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)

# 创建一个handler，
timestamp = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())
fh = logging.FileHandler(args.log_file + 'log_' + timestamp +'.txt')
fh.setLevel(logging.DEBUG)

# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# 定义handler的输出格式
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# 给logger添加handler
logger.addHandler(fh)
logger.addHandler(ch)


train_iter, val_iter, test_iter = load_data(args, logger)
logger.info('加载数据完成')
logger.info('load the model')
model = TextRNN(args)
torch.cuda.set_device(args.device)
model = model.cuda()
train(args, logger, train_iter, val_iter, model)
predict(test_iter, model, args, logger)




