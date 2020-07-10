import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop
from data_loader.data_one_domain import dataLoader
import copy


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, config, data_loader_C,
                      lr_scheduler=None, len_epoch=None):
        super().__init__(model, loss, metrics, optimizer, config)
        self.config = config
        # self.data_loader = data_loader

        self.aq_pos = ['dongsi_aq', 'tiantan_aq', 'guanyuan_aq', 'wanshouxigong_aq', 'aotizhongxin_aq', 'nongzhanguan_aq',
                      'wanliu_aq', 'fengtaihuayuan_aq', 'qianmen_aq', 'yongdingmennei_aq', 'xizhimenbei_aq',
                      'nansanhuan_aq',
                      'dongsihuan_aq']

        data_loader = []
        valid_loader = []

        for aq in self.aq_pos:
            data_loader.append(dataLoader(attribute_feature=["PM25", "PM10"],
                                       label_feature=["PM25", "PM10"],
                                       station=aq,
                                       mode="train",
                                       path="data/single.csv",
                                       predict_time_len=1,
                                       encoder_time_len=24,
                                       batch_size=128,
                                       num_workeres=1,
                                       shuffle=False))

            valid_loader.append(dataLoader(attribute_feature=["PM25", "PM10"],
                                             label_feature=["PM25", "PM10"],
                                             station=aq,
                                             mode="valid",
                                             path="data/single.csv",
                                             predict_time_len=1,
                                             encoder_time_len=24,
                                             batch_size=128,
                                             num_workeres=1,
                                             shuffle=False))



        self.data_loader = data_loader
        self.valid_loader = valid_loader

        if len_epoch is None:
            # epoch-based training
            # self.len_epoch = len(data_loader)
            self.len_epoch = 54
        else:
            # iteration-based training
            # self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.do_validation = True
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(config['data_loader']['args']['batch_size']))

        print(len(self.optimizer.param_groups))
        for i, param_group in enumerate(self.optimizer.param_groups):
            print(i)

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """

        self.model.train()

        total_loss = 0
        total_pm25_loss = 0
        total_pm10_loss = 0

        total_metrics = np.zeros(len(self.metrics))
        length = 0
        for i in range(len(self.aq_pos)):
            length += len(self.data_loader[i])
            for batch_idx, (feature, target) in enumerate(self.data_loader):
                feature, target = feature.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(feature)
                mse_loss = self.loss(output, target)

                pm25_predict, pm10_predict = torch.chunk(output, 2, dim=1)
                pm25_target, pm10_target = torch.chunk(target, 2, dim=1)
                pm25_loss = self.loss(pm25_predict, pm25_target)
                pm10_loss = self.loss(pm10_predict, pm10_target)
                total_pm25_loss += pm25_loss.item()
                total_pm10_loss += pm10_loss.item()


                loss = mse_loss
                l2_reg = torch.tensor(0.0).to(self.device)
                if self.config['trainer']['l2_regularization']:
                    for param in self.model.parameters():
                        l2_reg += torch.norm(param, p=2)
                    loss = loss + self.config['trainer']['l2_lambda'] * l2_reg

                loss.backward()
                # torch.nn.utils.clip_grad_norm(self.model.parameters(), max_norm=self.config['trainer']['clip_max_norm'])
                self.optimizer.step()

                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.writer.add_scalar('loss', mse_loss.item())
                total_loss += mse_loss.item()
                total_metrics += self._eval_metrics(output, target)

                if batch_idx % self.log_step == 0:
                    self.logger.debug('Train Epoch: {} {} Loss: {:.6f} mse: {:.6f} l2_reg: {:.6f}'.format(
                        epoch,
                        self._progress(batch_idx),
                        loss.item(),
                        mse_loss.item(),
                        l2_reg.item()
                    ))

        log = {
            'loss': total_loss / length,
            'rmse loss': np.sqrt(total_loss / length),
            'rmse_pm25_loss': np.sqrt(total_pm25_loss / length),
            'rmse_pm10_loss': np.sqrt(total_pm10_loss / length)
        }

        if self.do_validation:
            print('do_validation')
            val_log = self._valid_epoch(epoch)
            log.update(val_log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step(val_log['val_loss'])

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()

        total_val_loss = 0
        total_pm25_loss = 0
        total_pm10_loss = 0

        total_val_metrics = np.zeros(len(self.metrics))
        length = 0
        with torch.no_grad():
            for i in range(len(self.aq_pos)):
                length += len(self.valid_loader_C[i])
                for batch_idx, (feature, target) in enumerate(self.valid_loader[i]):
                    feature, target = feature.to(self.device), target.to(self.device)

                    output= self.model(feature, feature)
                    loss = self.loss(output, target)

                    pm25_predict, pm10_predict = torch.chunk(output, 2, 1)
                    pm25_target, pm10_target = torch.chunk(target, 2, 1)
                    pm25_loss = self.loss(pm25_predict, pm25_target)
                    pm10_loss = self.loss(pm10_predict, pm10_target)
                    total_pm25_loss += pm25_loss
                    total_pm10_loss += pm10_loss

                    self.writer.set_step((epoch - 1) * length + batch_idx, 'valid')
                    self.writer.add_scalar('loss', loss.item())
                    total_val_loss += loss.item()
                    total_val_metrics += self._eval_metrics(output, target)

        print('valid data length: ', length)
        return {
            'val_loss': total_val_loss / length,
            'val_rmse_loss': np.sqrt(total_val_loss / length),
            'val_pm25_rmse': np.sqrt(total_pm25_loss / length),
            'val_pm10_rmse': np.sqrt(total_pm10_loss / length)
        }

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'

        current = batch_idx
        total = self.len_epoch * len(self.aq_pos)
        return base.format(current, total, 100.0 * current / total)
