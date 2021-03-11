"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from dataprocess.DataLoader import TeethDataLoader
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='TeethModel', help='model name [default: TeethModel]')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch',  default=200, type=int, help='Epoch to run [default: 200]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--step_size', type=int,  default=20, help='Decay step for lr decay [default: every 20 epochs]')
    parser.add_argument('--lr_decay', type=float,  default=0.5, help='Decay rate for lr decay [default: 0.5]')

    return parser.parse_args()

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)

    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)

    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = 'data_1000/teeth_data/'

    TRAIN_DATASET = TeethDataLoader(split='train')
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size,shuffle=True, num_workers=4)
    TEST_DATASET = TeethDataLoader(split='test')
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size,shuffle=False, num_workers=4)
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" %  len(TEST_DATASET))

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/Net_utils.py', str(experiment_dir))

    classifier = MODEL.get_model().cuda()
    criterion = MODEL.get_loss().cuda()


    def weights_init(m): # 一些初始化参数吧
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0

    for epoch in range(start_epoch,args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        mean_correct = []
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x,momentum))

        '''learning one epoch'''
        for i, data in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            points, feature = data
            points = points.data.numpy()
            #取出第一维度CO的特征
            B, N, _ = feature.shape
            feature = feature[:,:,0].view(B,N,1)
            points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
            points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3]) # 随机抖动
            points = torch.Tensor(points)
            points, feature = points.float().cuda(),feature.float().cuda()
            points = points.transpose(2, 1)
            optimizer.zero_grad()
            classifier = classifier.train()
            feature_pred= classifier(points)
            loss = criterion(feature_pred, feature)
            mean_correct.append(loss)
            loss.backward()
            optimizer.step()
        train_loss = np.mean(mean_correct)
        log_string('Train loss is: %.5f' % train_loss)

        with torch.no_grad():
            test_metrics = {}
            loss_list = []

            for batch_id, (points, feature) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                cur_batch_size, NUM_POINT, _ = points.size()
                points, feature = points.float().cuda(), feature.float().cuda()
                points = points.transpose(2, 1)
                classifier = classifier.eval()
                feature_pred, _ = classifier(points)
                cur_pred_val = feature_pred.cpu().data.numpy()
                cur_pred_val_logits = cur_pred_val
                cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
                feature = feature.cpu().data.numpy()
                loss = torch.nn.functional.l1_loss(feature_pred, feature)
                loss_list.append(loss)

            mean_loss = np.mean(loss_list)
            test_metrics['loss'] = mean_loss

        log_string('Epoch %d test loss: %f' % (epoch+1, test_metrics['loss']))
        if (epoch == args.epoch):
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s'% savepath)
            state = {
                'epoch': epoch,
                'train_loss': train_loss,
                'test_loss': test_metrics['loss'],
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')
        global_epoch+=1

if __name__ == '__main__':
    args = parse_args()
    main(args)

