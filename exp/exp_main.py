from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import xPatch
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import math

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'xPatch': xPatch,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    # MSE and MAE criterion
    def _select_criterion(self):
        mse_criterion = nn.MSELoss()
        mae_criterion = nn.L1Loss()
        return mse_criterion, mae_criterion

    def _select_forecast_loss(self, mse_criterion, mae_criterion):
            loss_type = getattr(self.args, "train_loss_type", "mae")

            if loss_type == "mse":
                return mse_criterion
            elif loss_type == "mae":
                return mae_criterion
            else:
                raise ValueError("Unsupported train_loss_type: {}".format(loss_type))

    def _select_vali_loss(self, mse_criterion, mae_criterion):
        loss_type = getattr(self.args, "vali_loss_type", "mae")

        if loss_type == "mse":
            return mse_criterion
        elif loss_type == "mae":
            return mae_criterion
        else:
            raise ValueError("Unsupported vali_loss_type: {}".format(loss_type))

    def _get_model_for_aux(self):
        """
        Return the actual model object when DataParallel is used.
        """
        return self.model.module if hasattr(self.model, "module") else self.model

    def _get_aux_loss(self):
        """
        Optional auxiliary loss, e.g. CCM cluster loss.

        The model can implement:
            get_aux_loss()

        If the model does not provide it, return 0.0.
        """
        model_for_aux = self._get_model_for_aux()

        if not hasattr(model_for_aux, "get_aux_loss"):
            return 0.0

        aux_loss = model_for_aux.get_aux_loss()

        if aux_loss is None:
            return 0.0

        return aux_loss

    def _get_raw_cluster_loss(self):
        model_for_aux = self.model.module if hasattr(self.model, "module") else self.model

        if not hasattr(model_for_aux, "get_raw_cluster_loss"):
            return 0.0

        raw_loss = model_for_aux.get_raw_cluster_loss()

        if raw_loss is None:
            return 0.0

        return raw_loss

    def vali(self, vali_data, vali_loader, criterion, is_test=True):
        total_loss = []
        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp],
                    dim=1
                ).float().to(self.device)

                # encoder - decoder
                outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # if train-style validation, use ratio to scale the prediction
                use_loss_ratio = getattr(self.args, "use_loss_ratio", True)

                if (not is_test) and use_loss_ratio:
                    self.ratio = np.array([
                        -1 * math.atan(i + 1) + math.pi / 4 + 1
                        for i in range(self.args.pred_len)
                    ])
                    self.ratio = torch.tensor(self.ratio).unsqueeze(-1).to(self.device)

                    pred = outputs * self.ratio
                    true = batch_y * self.ratio
                else:
                    pred = outputs
                    true = batch_y

                loss = criterion(pred, true)
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        mse_criterion, mae_criterion = self._select_criterion()

        # train_times = [] # For computational cost analysis
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            # train_time = 0 # For computational cost analysis

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp],
                    dim=1
                ).float().to(self.device)

                # encoder - decoder
                # temp = time.time() # For computational cost analysis
                outputs = self.model(batch_x)

                model_for_aux = self.model.module if hasattr(self.model, "module") else self.model


                if i == 0 and epoch % 5 == 0:
                    if hasattr(model_for_aux, "net"):
                        net = model_for_aux.net

                        if hasattr(net, "ccm_prob") and net.ccm_prob is not None:
                            prob = net.ccm_prob.detach()

                            print("\n[CCM Diagnose] epoch:", epoch + 1)
                            print("prob mean over batch [C, K]:")
                            print(prob.mean(dim=0).cpu())

                            print("prob std over batch [C, K]:")
                            print(prob.std(dim=0).cpu())

                            print("prob cluster usage mean [K]:")
                            print(prob.mean(dim=(0, 1)).cpu())

                            print("prob max mean:")
                            print(prob.max(dim=-1).values.mean().item())

                        if hasattr(net, "ccm_sim_matrix") and net.ccm_sim_matrix is not None:
                            S = net.ccm_sim_matrix.detach()
                            B, C, _ = S.shape

                            eye = torch.eye(C, device=S.device, dtype=torch.bool).unsqueeze(0)
                            off_diag = S.masked_select(~eye.expand(B, C, C))

                            print("similarity S mean/std/min/max off-diag:")
                            print(
                                off_diag.mean().item(),
                                off_diag.std().item(),
                                off_diag.min().item(),
                                off_diag.max().item()
                            )

                            print("similarity S mean matrix [C, C]:")
                            print(S.mean(dim=0).cpu())
                # train_time += time.time() - temp # For computational cost analysis

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # Arctangent loss with weight decay
                forecast_criterion = self._select_forecast_loss(mse_criterion, mae_criterion)
                use_loss_ratio = getattr(self.args, "use_loss_ratio", True)

                if use_loss_ratio:
                    self.ratio = np.array([
                        -1 * math.atan(j + 1) + math.pi / 4 + 1
                        for j in range(self.args.pred_len)
                    ])
                    self.ratio = torch.tensor(self.ratio).unsqueeze(-1).to(self.device)

                    pred_for_loss = outputs * self.ratio
                    true_for_loss = batch_y * self.ratio
                else:
                    pred_for_loss = outputs
                    true_for_loss = batch_y

                # Main forecasting loss
                pred_loss = forecast_criterion(pred_for_loss, true_for_loss)
                loss = pred_loss

                aux_loss = self._get_aux_loss()
                raw_cluster_loss = self._get_raw_cluster_loss()

                aux_loss_value = 0.0
                raw_cluster_loss_value = 0.0

                if not isinstance(aux_loss, float):
                    loss = loss + aux_loss
                    aux_loss_value = aux_loss.item()

                if not isinstance(raw_cluster_loss, float):
                    raw_cluster_loss_value = raw_cluster_loss.item()
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f} pred: {3:.7f} aux: {4:.7f} raw_cluster: {5:.7f}".format(
                            i + 1,
                            epoch + 1,
                            loss.item(),
                            pred_loss.item(),
                            aux_loss_value,
                            raw_cluster_loss_value
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            # train_times.append(train_time / len(train_loader)) # For computational cost analysis
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            print(
                "Last batch loss details | pred: {:.7f} aux: {:.7f} raw_cluster: {:.7f}".format(
                    pred_loss.item(),
                    aux_loss_value,
                    raw_cluster_loss_value
                )
            )

            train_loss = np.average(train_loss)

            # Validation uses forecasting loss only.
            # Do not add auxiliary cluster loss here, otherwise early stopping may be misled.
            vali_criterion = self._select_vali_loss(mse_criterion, mae_criterion)

            # If use_loss_ratio=0, this is unweighted validation loss.
            # If use_loss_ratio=1 and is_test=False, this keeps original xPatch validation style.
            vali_loss = self.vali(vali_data, vali_loader, vali_criterion, is_test=False)

            # test_loss is only for logging, always unweighted MSE.
            test_loss = self.vali(test_data, test_loader, mse_criterion, is_test=True)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1,
                    train_steps,
                    train_loss,
                    vali_loss,
                    test_loss
                )
            )

            early_stopping(vali_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            # print('Alpha:', self.model.decomp.ma.alpha) # Print the learned alpha
            # print('Beta:', self.model.decomp.ma.beta)   # Print the learned beta

        # print("Training time: {}".format(np.sum(train_times) / len(train_times))) # For computational cost analysis
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        os.remove(best_model_path)

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))
            )

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # test_time = 0 # For computational cost analysis
        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp],
                    dim=1
                ).float().to(self.device)

                # encoder - decoder
                # temp = time.time() # For computational cost analysis
                outputs = self.model(batch_x)
                # test_time += time.time() - temp # For computational cost analysis

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        # print("Inference time: {}".format(test_time / len(test_loader))) # For computational cost analysis
        preds = np.array(preds)
        trues = np.array(trues)

        # preds = np.concatenate(preds, axis=0) # without the "drop-last" trick
        # trues = np.concatenate(trues, axis=0) # without the "drop-last" trick

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, rse, corr]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)

        return
