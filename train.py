# coding = utf-8
import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import traceback
import csv
from math import sqrt

device_count = torch.cuda.device_count()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def weighted_mse_loss(input, target, w):
    weight = torch.ones_like(target).to(device)
    for i in range(target.size(0)):
        if target[i] == 1:
            weight[i] = sqrt(w)
    input = input.unsqueeze(-1)
    target = target.unsqueeze(-1)
    loss = torch.sum(weight * (input - target) ** 2) / input.size()[0]
    loss /= torch.sum(weight)
    return loss


def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()
        
    if device_count > 1:
        model = nn.DataParallel(model)

    # cnn
    # Adam 优化
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=0.601)
    # torch.save(model, '1.pkl')
    steps = 0
    best_acc = 0
    last_step = 0
    # model.train()
    args.load_best_model=0
    


   
    for epoch in range(1, args.epochs+1): 
        print('\nEpoch:%s\n'%epoch)
        #训练开始

        model.train()
        for batch in train_iter:
           
            feature1, feature2, target, pairid = batch.issue1, batch.issue2, batch.label, batch.pairid
            with torch.no_grad():
                feature1.t_(), feature2.t_(), target.sub_(1), pairid.t_()# batch first, index align
            if args.cuda:
                feature1, feature2, target, pairid = feature1.cuda(), feature2.cuda(), target.cuda(), pairid.cuda()

            optimizer.zero_grad()
            # print(feature1.data)
            
            logit = model(feature1, feature2)
            # print(logit.data)
            # target = target.type(torch.cuda.FloatTensor)
            target = target.type(torch.FloatTensor).to(device)
            # print(target.data)
            # print('logit vector', logit.size())
            # print('target vector', target.size())
            loss_list = []
            length = len(target.data)
            for i in range(length):
                a = logit.data[i]
                b = target.data[i]
                loss_list.append(float(0.5*(b-a)*(b-a)))

            # print(loss_list)
            # loss = autograd.Variable(torch.cuda.FloatTensor(loss_list), requires_grad=True)
            # loss.backward(torch.FloatTensor([64*[1]]))
            # criterion = nn.MSELoss()
            # loss =criterion(logit, target)
            loss=weighted_mse_loss(logit, target,args.weight)
            # print(loss.grad)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                # print('\n')
                # corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                corrects = 0 # (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                for item in loss_list:
                    # print(item)
                    # print(type(item))
                    if item <= 0.125:
                        corrects += 1
                accuracy = 100.0 * corrects/batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps, 
                                                                             loss.item(), 
                                                                             accuracy,
                                                                             corrects,
                                                                             batch.batch_size))
                #
            if steps % 45 == 0:#rgs.test_interval == 0:
                # pass
                #
                dev_acc = eval(dev_iter, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
                #
            elif steps % args.save_interval == 0:
                # print('save loss: %s' %str(loss.data))
                save(model, args.save_dir, 'snapshot', steps)
        torch.cuda.empty_cache()


def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature1, feature2, target = batch.issue1, batch.issue2, batch.label
        with torch.no_grad():
            feature1.t_(), feature2.t_(), target.sub_(1)  # batch first, index align
        if args.cuda:
            feature1, feature2, target = feature1.cuda(), feature2.cuda(), target.cuda()

        logit = model(feature1, feature2)
        target = target.type(torch.FloatTensor)
        loss_list = []
        length = len(target.data)
        for i in range(length):
            a = logit.data[i]
            b = target.data[i]
            loss_list.append(float(0.5*(b-a)*(b-a)))
        corrects = 0 # (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        for item in loss_list:
            avg_loss += item 
            if item <= 0.125:
                 corrects += 1
        accuracy = 100.0 * float(corrects)/batch.batch_size 
    size = float(len(data_iter.dataset))
    avg_loss /= size
    accuracy = 100.0 * float(corrects)/size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))
    return accuracy

def eval_test(data_iter, model, args,prefix):
    to_path = './TSE/dataAll/'+prefix+'/'+prefix
    result_path = './TSE/dataAll/'+prefix
    model.eval()
    corrects, avg_loss = 0, 0
    f1_fenmu = 0
    f1_tp = 0
    loss_list = []
    id_list = []
    sim_list = []
    tar_list = []
    for batch in data_iter:
        feature1, feature2, target, pairid = batch.issue1, batch.issue2, batch.label, batch.pairid
        with torch.no_grad():
            feature1.t_(), feature2.t_(), target.sub_(1), pairid.t_()  # batch first, index align
        # feature1.data.t_(), feature2.data.t_(), target.data.sub_(1), pairid.data.t_()  # batch first, index align
        if args.cuda:
            feature1, feature2, target, pairid = feature1.cuda(), feature2.cuda(), target.cuda(), pairid.cuda()

        logit = model(feature1, feature2)
        target = target.type(torch.FloatTensor)
        pairid = pairid.type(torch.FloatTensor)
        length = len(target.data)
        for i in range(length):
            a = logit.data[i]
            b = target.data[i]
            sim_list.append(a)
            tar_list.append(b)
            id_list.append(int(pairid.data[i]))
            if a >= 0.5:
                f1_fenmu += 1
                if b == 1:
                    f1_tp += 1
            loss_list.append(float(0.5*(b-a)*(b-a)))
         # (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        # f1
        
    print('f1:{:.6f}\n'.format(float(f1_tp)/float(f1_fenmu)))
    for item in loss_list:
        avg_loss += item 
        if item <= 0.125:
            corrects += 1
    size = float(len(data_iter.dataset))
    avg_loss /= size
    accuracy = 100.0 * float(corrects)/size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))
    tmp = pd.DataFrame()
    # print(sim_list)
    # print('+++')
    # print(sim_list.cpu().numpy())
    # print('===')
    # print([i.data.cpu().squeeze() for i in sim_list])
    tmp['sim'] = [float(i) for i in sim_list]
    tmp['label'] = [float(i) for i in tar_list]
    tmp['pair_id'] = [int(i) for i in id_list]
    
    tmp.to_csv(to_path+'_sim'+'.csv')
    res = []
    cnt = 0
    for i,r in tmp.iterrows():
        if i >= 0:
            if (r['sim'] >= 0.5) & (r['label'] == 1):
                cnt += 1
                res.append(1)
            elif (r['sim'] < 0.5) & (r['label'] == 0):
                cnt += 1
                res.append(1)
            else:
                res.append(0)
    precision = 0.0
    # cnt_true = 0
    # for i in range(len(res)):
    #     if res[i] == 1:
    #         cnt_true += 1
    #         if res[i] == list(tmp['label'])[i]:    
    #             precision += 1
    # t = precision
    # precision = t / float(sum(res))
    
    cnt_true = 0
    for i in range(len(res)):
       if res[i] == list(tmp['label'])[i]:
      # if res[i] == 1:
            cnt_true += 1
            if res[i] == 1:
           # if res[i] == list(tmp['label'])[i]:   
                precision += 1
    t = precision
    precision = t / (float(cnt_true))

    recall = t /sum(list(tmp['label']))
    f1 = 2*precision*recall / (precision + recall)
    print('precision' + ' ' + '%f' % (precision))
    print('recall' + ' ' + '%f' % (recall))
    print('accuracy' + ' ' + '%f' % (float(cnt) / len(tmp)))
    print('f1' + ' ' + '%f' % (f1))
    print('TP' + ' ' + '%f' % t)
    print('FP' + ' ' + '%f' % ((cnt_true - t)))
    print('TN' + ' ' + '%f' % (cnt - t))
    print('FN' + ' ' + '%f' % (sum(list(tmp['label'])) - t))
    with open(result_path + '/result.txt', 'a') as f:
        print(f'precision: {precision:.2f}', file=f)
        print(f'recall: {recall:.2f}', file=f)
        print(f'accuracy: {float(cnt)/len(tmp):.2f}', file=f)
        print(f'f1: {f1:.2f}', file=f)
        print(f'TP: {t:.2f}', file=f)
        print(f'FP: {(cnt_true - t):.2f}', file=f)
        print(f'TN: {(cnt - t):.2f}', file=f)
        print(f'FN: {(sum(list(tmp["label"])) - t):.2f}', file=f)


    return accuracy

def predict(line, model, issue1_field, issue2_field, label_field, cuda_flag):
    # assert isinstance(text, str)
    model.eval()
    # text = text_field.tokenize(text)
    issue1 = issue1_field.preprocess(line.split(',')[1])
    issue2 = issue2_field.preprocess(line.split(',')[2])
    issue1 = [[issue1_field.vocab.stoi[x] for x in issue1]]
    issue2 = [[issue2_field.vocab.stoi[x] for x in issue2]]
    # text = text_field.preprocess(text)
    # text = [[text_field.vocab.stoi[x] for x in text]]
    # x = text_field.tensor_type(text)
    # x = autograd.Variable(x, volatile=True)
    
    i1 = issue1_field.tensor_type(issue1)
    i1 = autograd.Variable(i1, volatile=True)
    
    i2 = issue2_field.tensor_type(issue2)
    i2 = autograd.Variable(i2, volatile=True)
    if cuda_flag:
        i1 = i1.cuda()
        i2 = i2.cuda()
    # print(x)
    # print(i1.data)
    # print(i2.data)
    output = model(i1, i2)
    return (output.data[0])


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
