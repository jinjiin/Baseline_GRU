import argparse
import torch
from tqdm import tqdm
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from data_loader.data_loader import dataLoader
import torch.nn.functional as F
import numpy as np


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    aq_pos = ['dongsi_aq', 'tiantan_aq', 'guanyuan_aq', 'wanshouxigong_aq', 'aotizhongxin_aq',
                   'nongzhanguan_aq',
                   'wanliu_aq', 'fengtaihuayuan_aq', 'qianmen_aq', 'yongdingmennei_aq', 'xizhimenbei_aq',
                   'nansanhuan_aq',
                   'dongsihuan_aq']
    data_loader = []
    for aq in aq_pos:
        data_loader.append(dataLoader(
            attribute_feature=["temperature"],
            label_feature=["temperature"],
            station=aq, mode="valid",
            path="/lfs1/users/jbyu/QA_inference/data/single.csv",
            predict_time_len=1,
            encoder_time_len=24,
            batch_size=128,
            num_workeres=1,
            shuffle=False,
            embedding=False))

    # build model architecture
    model = config.initialize('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    entity = list(np.loadtxt('data/entity.csv', delimiter=','))
    entity = torch.Tensor(entity).to(device)

    total_loss = 0.0
    total_mse_loss = 0.0
    length = 0

    with torch.no_grad():
        for i in range(len(aq_pos)):

            for batch_idx, (data, _, target) in enumerate(data_loader[i]):
                length += data.shape[0]
                # print(batch_idx)
                data, target = data.to(device), target.to(device)
                output = model(data)

                """
                # classifity
                target = target.squeeze()
                p = F.softmax(output, dim=1)
                p = torch.argmax(p, dim=1)
                cur_loss = F.mse_loss(entity[p], entity[target], size_average=False)
                print(cur_loss)
                total_mse_loss += cur_loss.item()
                """

                # regression
                cur_loss = F.mse_loss(output, target, size_average=False)
                print(output.shape, target.shape, cur_loss)
                total_mse_loss += cur_loss.item()
                loss = loss_fn(output, target)
                total_loss += loss.item()



    log = {'cross_entropy loss': total_loss / length,
           'mse loss': total_mse_loss / length}
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser(args)
    main(config)
