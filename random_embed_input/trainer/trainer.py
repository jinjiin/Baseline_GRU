import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, config, data_loader, valid_data_loader,
                 lr_scheduler=None, len_epoch=None):
        super().__init__(model, loss, metrics, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.data_loader = data_loader
        self.valid_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))


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
        total_metrics = np.zeros(len(self.metrics))
        length = len(self.data_loader)
        total_pm25_loss = 0
        total_pm10_loss = 0

        for batch_idx, (data,  target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            # print(output.shape, target.shape)
            target = target.squeeze()
            model_loss = self.loss(output, target)

            # print(output.shape, target.shape)
            pm25_predict, pm10_predict = torch.chunk(output, 2, dim=1)
            pm25_target, pm10_target = torch.chunk(target, 2, dim=1)
            pm25_loss = self.loss(pm25_predict, pm25_target)
            pm10_loss = self.loss(pm10_predict, pm10_target)
            total_pm25_loss += pm25_loss.item()
            total_pm10_loss += pm10_loss.item()

            l2_reg = torch.tensor(0.0).to(self.device)
            if self.config['trainer']['l2_regularization']:
                for param in self.model.parameters():
                    l2_reg += torch.norm(param, p=2)
                loss = model_loss + self.config['trainer']['l2_lambda'] * l2_reg
            else:
                loss = model_loss

            loss.backward()
            # torch.nn.utils.clip_grad_norm(self.model.parameters(), 1)
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item()
            # total_metrics += self._eval_metrics(output, target)

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} model_loss: {:.6f} pm25_loss {:.6f} pm10_loss {:.6f} l2_loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                    model_loss.item(),
                    pm25_loss.item(),
                    pm10_loss.item(),
                    l2_reg.item()))

            if batch_idx == self.len_epoch:
                break

        log = {
            'loss': total_loss / length,
            'rmse_pm25_loss': np.sqrt(total_pm25_loss / length),
            'rmse_pm10_loss': np.sqrt(total_pm10_loss / length)
        }


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
        total_val_metrics = np.zeros(len(self.metrics))
        length = len(self.valid_loader)
        total_pm25_loss = 0
        total_pm10_loss = 0

        with torch.no_grad():

            for batch_idx, (data, target) in enumerate(self.valid_loader):
                data, target = data.to(self.device), target.to(self.device)
                target = target.squeeze()
                output = self.model(data)
                # print(output)
                loss = self.loss(output, target)
                pm25_predict, pm10_predict = torch.chunk(output, 2, dim=1)
                pm25_target, pm10_target = torch.chunk(target, 2, dim=1)
                pm25_loss = self.loss(pm25_predict, pm25_target)
                pm10_loss = self.loss(pm10_predict, pm10_target)
                total_pm25_loss += pm25_loss.item()
                total_pm10_loss += pm10_loss.item()

                self.writer.set_step((epoch - 1) * len(self.valid_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(output, target)

        return {
            'val_loss': total_val_loss / length,
            'val_rmse_loss':np.sqrt(total_val_loss / length),
            'val_pm25_loss': np.sqrt(total_pm25_loss / length),
            'val_pm10_loss': np.sqrt(total_pm10_loss / length)
        }

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
