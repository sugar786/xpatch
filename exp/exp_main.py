from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import xPatch
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.ccm_diagnosis import (
    ccm_diagnose_tensors,
    save_ccm_visuals,
    save_embedding_pca,
)

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
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

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
        return self.model.module if hasattr(self.model, "module") else self.model

    def _get_aux_loss(self):
        model_for_aux = self._get_model_for_aux()

        if not hasattr(model_for_aux, "get_aux_loss"):
            return 0.0

        aux_loss = model_for_aux.get_aux_loss()

        if aux_loss is None:
            return 0.0

        return aux_loss

    def _get_raw_cluster_loss(self):
        model_for_aux = self._get_model_for_aux()

        if not hasattr(model_for_aux, "get_raw_cluster_loss"):
            return 0.0

        raw_loss = model_for_aux.get_raw_cluster_loss()

        if raw_loss is None:
            return 0.0

        return raw_loss

    def _get_raw_cluster_loss_s(self):
        model_for_aux = self._get_model_for_aux()

        if not hasattr(model_for_aux, "get_raw_cluster_loss_s"):
            return 0.0

        raw_loss = model_for_aux.get_raw_cluster_loss_s()

        if raw_loss is None:
            return 0.0

        return raw_loss

    def _get_raw_cluster_loss_t(self):
        model_for_aux = self._get_model_for_aux()

        if not hasattr(model_for_aux, "get_raw_cluster_loss_t"):
            return 0.0

        raw_loss = model_for_aux.get_raw_cluster_loss_t()

        if raw_loss is None:
            return 0.0

        return raw_loss

    def _print_single_ccm_diag(self, name, S, prob, membership):
        diag = ccm_diagnose_tensors(S, prob, membership)

        print("[CCM Diagnose - {}]".format(name))
        print("S offdiag mean/std/min/max: {:.6f} {:.6f} {:.6f} {:.6f}".format(
            diag["S_off_mean"],
            diag["S_off_std"],
            diag["S_off_min"],
            diag["S_off_max"],
        ))
        print("S-P alignment corr: {:.6f}".format(diag["S_P_alignment_corr"]))
        print("P entropy: {:.6f}, normalized: {:.6f}".format(
            diag["P_entropy"],
            diag["P_entropy_norm"],
        ))
        print("P max mean: {:.6f}".format(diag["P_max_mean"]))
        print("P usage:", diag["P_usage"])
        print("P balance gap: {:.6f}".format(diag["P_balance_gap"]))
        print("P dist mean/std: {:.6f} {:.6f}".format(
            diag["P_dist_mean"],
            diag["P_dist_std"],
        ))

        if "M_usage" in diag:
            print("Membership usage:", diag["M_usage"])

    def _save_single_ccm_visuals(
        self,
        S,
        prob,
        channel_emb,
        cluster_emb,
        path,
        epoch,
        name,
    ):
        save_dir = os.path.join(path, "ccm_visuals")
        prefix = "{}_epoch_{:03d}".format(name, epoch + 1)

        save_ccm_visuals(
            S=S,
            P=prob,
            save_dir=save_dir,
            prefix=prefix,
        )

        if (
            getattr(self.args, "ccm_save_pca", 0)
            and channel_emb is not None
            and cluster_emb is not None
        ):
            save_embedding_pca(
                channel_emb=channel_emb.detach(),
                cluster_emb=cluster_emb.detach(),
                save_dir=save_dir,
                prefix=prefix,
            )

    def _maybe_diagnose_ccm(self, epoch, batch_idx, path, pred_loss, aux_loss):
        """
        Print and optionally save CCM diagnostics on the first batch of selected epochs.
        Supports both single-CCM and dual-CCM.
        """
        diag_interval = getattr(self.args, "ccm_diag_interval", 5)
        if diag_interval <= 0:
            return

        if batch_idx != 0:
            return

        if epoch % diag_interval != 0:
            return

        model_for_aux = self._get_model_for_aux()

        if not hasattr(model_for_aux, "net"):
            return

        net = model_for_aux.net

        print("\n[CCM Diagnose] epoch:", epoch + 1)

        if getattr(self.args, "use_dual_ccm", 0):
            # Seasonal CCM diagnosis
            if (
                hasattr(net, "ccm_prob_s") and net.ccm_prob_s is not None
                and hasattr(net, "ccm_sim_matrix_s") and net.ccm_sim_matrix_s is not None
            ):
                prob_s = net.ccm_prob_s.detach()
                S_s = net.ccm_sim_matrix_s.detach()
                M_s = net.ccm_membership_s.detach() if net.ccm_membership_s is not None else None

                self._print_single_ccm_diag("seasonal", S_s, prob_s, M_s)

                if getattr(self.args, "ccm_save_visuals", 0):
                    self._save_single_ccm_visuals(
                        S=S_s,
                        prob=prob_s,
                        channel_emb=net.ccm_channel_emb_s,
                        cluster_emb=net.ccm_cluster_emb_s,
                        path=path,
                        epoch=epoch,
                        name="seasonal",
                    )

            # Trend CCM diagnosis
            if (
                hasattr(net, "ccm_prob_t") and net.ccm_prob_t is not None
                and hasattr(net, "ccm_sim_matrix_t") and net.ccm_sim_matrix_t is not None
            ):
                prob_t = net.ccm_prob_t.detach()
                S_t = net.ccm_sim_matrix_t.detach()
                M_t = net.ccm_membership_t.detach() if net.ccm_membership_t is not None else None

                self._print_single_ccm_diag("trend", S_t, prob_t, M_t)

                if getattr(self.args, "ccm_save_visuals", 0):
                    self._save_single_ccm_visuals(
                        S=S_t,
                        prob=prob_t,
                        channel_emb=net.ccm_channel_emb_t,
                        cluster_emb=net.ccm_cluster_emb_t,
                        path=path,
                        epoch=epoch,
                        name="trend",
                    )

        else:
            # Single CCM diagnosis
            if (
                hasattr(net, "ccm_prob") and net.ccm_prob is not None
                and hasattr(net, "ccm_sim_matrix") and net.ccm_sim_matrix is not None
            ):
                prob = net.ccm_prob.detach()
                S = net.ccm_sim_matrix.detach()
                membership = net.ccm_membership.detach() if net.ccm_membership is not None else None

                self._print_single_ccm_diag("single", S, prob, membership)

                if getattr(self.args, "ccm_save_visuals", 0):
                    self._save_single_ccm_visuals(
                        S=S,
                        prob=prob,
                        channel_emb=net.ccm_channel_emb,
                        cluster_emb=net.ccm_cluster_emb,
                        path=path,
                        epoch=epoch,
                        name="single",
                    )

        if not isinstance(aux_loss, float):
            aux_value = aux_loss.item()
        else:
            aux_value = 0.0

        print("aux/pred ratio: {:.6f}".format(
            aux_value / (pred_loss.item() + 1e-8)
        ))

        raw_s = self._get_raw_cluster_loss_s()
        raw_t = self._get_raw_cluster_loss_t()

        raw_s_value = raw_s.item() if not isinstance(raw_s, float) else 0.0
        raw_t_value = raw_t.item() if not isinstance(raw_t, float) else 0.0

        if getattr(self.args, "use_dual_ccm", 0):
            print("raw_cluster_s: {:.7f} raw_cluster_t: {:.7f}".format(
                raw_s_value,
                raw_t_value,
            ))

    def vali(self, vali_data, vali_loader, criterion, is_test=True):
        total_loss = []
        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp],
                    dim=1
                ).float().to(self.device)

                outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

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

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp],
                    dim=1
                ).float().to(self.device)

                outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

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

                pred_loss = forecast_criterion(pred_for_loss, true_for_loss)
                loss = pred_loss

                aux_loss = self._get_aux_loss()
                raw_cluster_loss = self._get_raw_cluster_loss()
                raw_cluster_loss_s = self._get_raw_cluster_loss_s()
                raw_cluster_loss_t = self._get_raw_cluster_loss_t()

                aux_loss_value = 0.0
                raw_cluster_loss_value = 0.0
                raw_cluster_loss_s_value = 0.0
                raw_cluster_loss_t_value = 0.0

                if not isinstance(aux_loss, float):
                    loss = loss + aux_loss
                    aux_loss_value = aux_loss.item()

                if not isinstance(raw_cluster_loss, float):
                    raw_cluster_loss_value = raw_cluster_loss.item()

                if not isinstance(raw_cluster_loss_s, float):
                    raw_cluster_loss_s_value = raw_cluster_loss_s.item()

                if not isinstance(raw_cluster_loss_t, float):
                    raw_cluster_loss_t_value = raw_cluster_loss_t.item()

                self._maybe_diagnose_ccm(
                    epoch=epoch,
                    batch_idx=i,
                    path=path,
                    pred_loss=pred_loss,
                    aux_loss=aux_loss,
                )

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f} pred: {3:.7f} aux: {4:.7f} raw_cluster: {5:.7f} raw_s: {6:.7f} raw_t: {7:.7f}".format(
                            i + 1,
                            epoch + 1,
                            loss.item(),
                            pred_loss.item(),
                            aux_loss_value,
                            raw_cluster_loss_value,
                            raw_cluster_loss_s_value,
                            raw_cluster_loss_t_value,
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            print(
                "Last batch loss details | pred: {:.7f} aux: {:.7f} raw_cluster: {:.7f} raw_s: {:.7f} raw_t: {:.7f}".format(
                    pred_loss.item(),
                    aux_loss_value,
                    raw_cluster_loss_value,
                    raw_cluster_loss_s_value,
                    raw_cluster_loss_t_value,
                )
            )

            train_loss = np.average(train_loss)

            vali_criterion = self._select_vali_loss(mse_criterion, mae_criterion)

            vali_loss = self.vali(vali_data, vali_loader, vali_criterion, is_test=False)

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

        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp],
                    dim=1
                ).float().to(self.device)

                outputs = self.model(batch_x)

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

        preds = np.array(preds)
        trues = np.array(trues)

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

        return
