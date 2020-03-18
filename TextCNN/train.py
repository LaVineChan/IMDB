import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd


def train(args, logger, train_iter, dev_iter, model):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    best_corrects = 0
    for epoch in tqdm(range(1, args.epoch + 1)):
        model.train()
        all_loss = 0
        for batch in train_iter:
            feature, target = batch.review_text, batch.is_spoiler
            with torch.no_grad():
                feature = feature.t_()
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)

            target = target.to(dtype=torch.long)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            steps += 1
            all_loss += loss.item()
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/batch.batch_size
                logger.info('Batch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,loss.item(), accuracy, corrects, batch.batch_size))
            # if steps % args.test_interval == 0:
        dev_corrects = eval(dev_iter, model, args, logger)
        if dev_corrects > best_corrects:
            best_corrects = dev_corrects
            logger.info('the epoch: {} step: {}, corrects: {:.4f}'.format(epoch, steps, best_corrects))
            # save(model, args.save_dir, epoch, 'best', steps, logger)
        logger.info('the all loss of epoch {} are {}'.format(epoch, all_loss))

def eval(data_iter, model, args, logger):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.review_text, batch.is_spoiler
        with torch.no_grad():
           feature = feature.t_()
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        target = target.to(dtype=torch.long)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.item()
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0000 * corrects/size
    logger.info('Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, accuracy, corrects, size))
    return corrects


def predict(test_iter, model, args, logger):
    logger.info("predict the result")
    if args.cuda:
        model.cuda()
    pred_result = torch.tensor([], dtype=torch.long).cuda()
    model.eval()
    for idx, batch in enumerate(test_iter):
        feature = batch.review_text
        with torch.no_grad():
            feature = feature.t_()
        if args.cuda:
            feature = feature.cuda()
        logit = model(feature)
        _, predicted = torch.max(logit, 1)
        pred_result = torch.cat((pred_result, predicted), 0)
    pred_result = pred_result.tolist()
    submission = pd.DataFrame({'id': range(len(pred_result)), 'pred': pred_result})
    submission['id'] = submission['id']
    submission.to_csv("./textCNN_submission_0.001.csv", index=False, header=False)



def save(model, save_dir, epoch, save_prefix, steps, logger):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_epoch_{}_steps_{}.pt'.format(save_prefix, epoch,  steps)
    logger.info("save the model")
    torch.save(model.state_dict(), save_path)
