import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from logging import getLogger

from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss


# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)


class BnReluConv(nn.Module):
    def __init__(self, nb_filter, bn=False):
        super(BnReluConv, self).__init__()
        self.has_bn = bn
        self.bn1 = nn.BatchNorm2d(nb_filter)
        self.relu = torch.relu
        self.conv1 = conv3x3(nb_filter, nb_filter)

    def forward(self, x):
        if self.has_bn:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        return x


class ResidualUnit(nn.Module):
    def __init__(self, nb_filter, bn=False):
        super(ResidualUnit, self).__init__()
        self.bn_relu_conv1 = BnReluConv(nb_filter, bn)
        self.bn_relu_conv2 = BnReluConv(nb_filter, bn)

    def forward(self, x):
        residual = x
        out = self.bn_relu_conv1(x)
        out = self.bn_relu_conv2(out)
        out += residual  # short cut
        return out


class ResUnits(nn.Module):
    def __init__(self, residual_unit, nb_filter, repetations=1, bn=False):
        super(ResUnits, self).__init__()
        self.stacked_resunits = self.make_stack_resunits(residual_unit, nb_filter, repetations, bn)

    def make_stack_resunits(self, residual_unit, nb_filter, repetations, bn):
        layers = []
        for i in range(repetations):
            layers.append(residual_unit(nb_filter, bn))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stacked_resunits(x)
        return x


def INF3DH(B, H, W, D):
    return -torch.diag(torch.tensor(float("inf")).repeat(H), 0).unsqueeze(0).repeat(B * W * D, 1, 1)  # .cuda()


def INF3DW(B, H, W, D):
    return -torch.diag(torch.tensor(float("inf")).repeat(W), 0).unsqueeze(0).repeat(B * H * D, 1, 1)  # .cuda()


def INF3DD(B, H, W, D):
    return -torch.diag(torch.tensor(float("inf")).repeat(D), 0).unsqueeze(0).repeat(B * H * W, 1, 1)  # .cuda()


class CrissCrossAttention3D(nn.Module):
    """ Criss-Cross Attention Module 3D version, inspired by the 2d version, but 3D CC Module should mask out the overlapped elements twice!"""

    def __init__(self, in_dim, verbose=False):
        super(CrissCrossAttention3D, self).__init__()
        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=4)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.verbose = verbose
        self.INFH = INF3DH
        self.INFD = INF3DD

    def forward(self, x):
        # bcdhw -> bchwt
        # (B,C,H,W,T)
        m_batchsize, _, height, width, depth = x.size()
        query = self.query_conv(x)
        # (B,C/8,H,W,T)
        # (B,C/8,H,W,T) -> (B,W,T,C/8,H) -> (B*W*T,C/8,H) -> (B*W*T,H,C/8)
        # bchw > bwch, b*w*d-c-h > b*w*d-h-c
        query_H = query.permute(0, 3, 4, 1, 2).contiguous().view(m_batchsize * width * depth,
                                                                 -1,
                                                                 height).permute(0, 2, 1)
        # bchw > bhcw, b*h*d-c-w > b*h*d-w-c
        query_W = query.permute(0, 2, 4, 1, 3).contiguous().view(m_batchsize * height * depth, -1,
                                                                 width).permute(0, 2, 1)
        # bchwd > bwch, b*h*w-c-d > b*h*w-d-c
        query_D = query.permute(0, 2, 3, 1, 4).contiguous().view(m_batchsize * height * width, -1,
                                                                 depth).permute(0, 2, 1)

        key = self.key_conv(x)

        # bchw > bwch, b*w*d-c-h
        key_H = key.permute(0, 3, 4, 1, 2).contiguous().view(m_batchsize * width * depth, -1, height)
        # bchw > bhcw, b*h*d-c-w
        key_W = key.permute(0, 2, 4, 1, 3).contiguous().view(m_batchsize * height * depth, -1, width)
        key_D = key.permute(0, 2, 3, 1, 4).contiguous().view(m_batchsize * height * width, -1,
                                                             depth)  # b*h*w-c-d

        value = self.value_conv(x)
        value_H = value.permute(0, 3, 4, 1, 2).contiguous().view(m_batchsize * width * depth, -1,
                                                                 height)  # bchwd->bwdch
        value_W = value.permute(0, 2, 4, 1, 3).contiguous().view(m_batchsize * height * depth, -1,
                                                                 width)  # bchwd->bhdcw
        value_D = value.permute(0, 2, 3, 1, 4).contiguous().view(m_batchsize * height * width, -1,
                                                                 depth)  # bchwd->bhwcd

        # batch matrix-matrix
        inf_holder = self.INFH(m_batchsize, height, width, depth).to(x.device)  # > bw-h-h

        energy_H = torch.bmm(query_H, key_H) + inf_holder  # bwd-h-c, bwd-c-h > bwd-h-h
        energy_H = energy_H.view(m_batchsize, width, depth, height, height).permute(0, 1, 3, 2, 4)  # bwhdh

        #  b*h*d-w-c, b*h*d-c-w > b*h*d-w-w
        energy_W = torch.bmm(query_W, key_W)  # +self.INFW(m_batchsize, height, width, depth)
        energy_W = energy_W.view(m_batchsize, height, depth, width, width).permute(0, 3, 1, 2, 4)  # bwhdw

        #  b*h*w-d-c, b*h*w-c-d > b*h*w-d-d
        energy_D = (torch.bmm(query_D, key_D) + self.INFD(m_batchsize, height, width, depth).to(
            x.device)).view(m_batchsize, height, width, depth, depth).permute(0, 2, 1, 3, 4)  # bwhdd

        concate = self.softmax(torch.cat([energy_H, energy_W, energy_D], 4))  # bwhd*(h+w+d)

        # bhw(H+W) > bhwH, bwhH;
        att_H = concate[:, :, :, :, 0:height].permute(0, 1, 3, 2, 4).contiguous().view(m_batchsize * width * depth,
                                                                                       height, height)
        att_W = concate[:, :, :, :, height:height + width].permute(0, 2, 3, 1, 4).contiguous().view(
            m_batchsize * height * depth, width, width)
        att_D = concate[:, :, :, :, height + width:].permute(0, 2, 1, 3, 4).contiguous().view(
            m_batchsize * height * width, depth, depth)

        # p-c-h, p-h-h > p-c-h
        out_H = torch.bmm(value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, depth, -1, height).permute(0, 3, 4,
                                                                                                               1, 2)
        out_W = torch.bmm(value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, depth, -1, width).permute(0, 3, 1,
                                                                                                               4, 2)
        out_D = torch.bmm(value_D, att_D.permute(0, 2, 1)).view(m_batchsize, height, width, -1, depth).permute(0, 3, 1,
                                                                                                               2, 4)

        # print(out_H.size(),out_W.size())
        return self.gamma * (out_H + out_W + out_D) + x


class ST3DCCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, forward_expansion=4):
        super(ST3DCCBlock, self).__init__()
        inter_channels = in_channels // 4
        self.conv_in = nn.Sequential(nn.Conv3d(in_channels, inter_channels, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm3d(inter_channels),
                                     nn.ReLU(inplace=False))
        self.cca = CrissCrossAttention3D(inter_channels)
        self.conv_mid = nn.Sequential(nn.Conv3d(inter_channels, inter_channels, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm3d(inter_channels),
                                      nn.ReLU(inplace=False))

        self.conv_out = nn.Sequential(
            nn.Conv3d(in_channels + inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False),
            nn.Dropout3d(0.5),
            nn.Conv3d(out_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True))
        # self.feed_forward = nn.Sequential(nn.Linear(in_channels, in_channels * forward_expansion),
        #                                   nn.ReLU(inplace=False),
        #                                   nn.Linear(in_channels * forward_expansion, out_channels))

    def forward(self, x, recurrence=2):
        # (B,C,T,N,N)
        output = self.conv_in(x)
        # (B,C/4,T,N,N)
        output = output.permute(0, 1, 3, 4, 2)
        # (B,C/4,N,N,T)
        for i in range(recurrence):
            output = self.cca(output)
        # bchwd -> bcdhw
        output = output.permute(0, 1, 4, 2, 3)
        # (B,C/4,T,N,N)
        output = self.conv_mid(output)
        # (B,C/4,T,N,N)
        output = self.conv_out(torch.cat([x, output], 1))
        # (B,C,T,N,N)
        # output = self.feed_forward(output + x)
        return F.relu(output + x)


class MOMO(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.num_nodes = self.data_feature.get('num_nodes')
        self._scaler = self.data_feature.get('scaler')
        self.output_dim = config.get('output_dim')
        self.device = config.get('device', torch.device('cpu'))
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)

        self.p_interval = config.get('p_interval', 1)
        self.embed_dim = config.get('embed_dim')
        self.batch_size = config.get('batch_size')
        self.loss_p0 = config.get('loss_p0', 0.5)
        self.loss_p1 = config.get('loss_p1', 0.25)
        self.loss_p2 = config.get('loss_p2', 0.25)
        self.nb_residual_unit = config.get('nb_residual_unit',4)
        self.bn = config.get('bn',False)

        dis_mx = self.data_feature.get('adj_mx')
        self.conv3d_1 = nn.Sequential(nn.Conv3d(in_channels=1,out_channels=16,kernel_size=3,padding=1),
                                      nn.ReLU(inplace=False),
                                      nn.Conv3d(in_channels=16,out_channels=32,kernel_size=3,padding=1),
                                      nn.ReLU(inplace=False)
                                      )
        self.ST_Blocks = ST3DCCBlock(32, 32)
        self.embed = nn.Sequential(nn.Conv2d(in_channels=self.input_window*32, out_channels=128,kernel_size=1),
                                   nn.ReLU(inplace=False),
                                   nn.Conv2d(in_channels=128,out_channels=64,kernel_size=1),
                                   nn.ReLU(inplace=False))
        self.spablock = ResUnits(ResidualUnit, nb_filter=64, repetations=self.nb_residual_unit, bn=self.bn)
        self.output = nn.Sequential(nn.Conv2d(64, 16,kernel_size=1),
                                    nn.ReLU(inplace=False),
                                    nn.Conv2d(16, self.output_dim,kernel_size=1))

        # self.geo_adj = generate_geo_adj(dis_mx) \
        #     .repeat(self.batch_size * self.input_window, 1) \
        #     .reshape((self.batch_size, self.input_window, self.num_nodes, self.num_nodes)) \
        #     .to(self.device)
        #
        # self.GCN = GCN(self.num_nodes, self.embed_dim, self.device)
        #
        # # self.LSTM = nn.LSTM(2 * self.embed_dim, 2 * self.embed_dim)
        # self.LSTM = SLSTM(2 * self.embed_dim, 2 * self.embed_dim, self.device, self.p_interval)
        #
        # self.mutiLearning = MutiLearning(2 * self.embed_dim, self.device)

    def forward(self, batch):
        x = batch['X'].squeeze(dim=-1)
        # (B, T, N, N)
        x = x.unsqueeze(1)
        x = self.conv3d_1(x)
        x = self.ST_Blocks(x)
        x = x.reshape((x.shape[0], -1, self.num_nodes, self.num_nodes))
        x = self.embed(x)
        x = self.spablock(x)
        out = self.output(x)
        out = out.unsqueeze(dim=-1)
        # x_ge_embed = self.GCN(x, self.geo_adj[:x.shape[0], ...])
        # # (B, T, N, E)
        #
        # x_se_embed = self.GCN(x, self.semantic_adj)
        #
        # # (B, T, N, E)
        # x_embed = torch.cat([x_ge_embed, x_se_embed], dim=3)
        # # (B, T, N, 2E)
        # x_embed = x_embed.permute(1, 0, 2, 3)
        # # (T, B, N, 2E)
        # x_embed = x_embed.reshape((self.input_window, -1, 2 * self.embed_dim))
        # # (T,
        # # _, (h, _) = self.LSTM(x_embed)
        # # x_embed_pred = h[0].reshape((self.batch_size, -1, 2 * self.embed_dim))
        # x_embed_pred = self.LSTM(x_embed).reshape((x.shape[0], -1, 2 * self.embed_dim))
        # # (B, N, 2E)
        #
        # out = self.mutiLearning(x_embed_pred)

        return out

    def calculate_loss(self, batch):
        y_true = batch['y']  # (B, TO, N, N, 1)
        # y_in_true = torch.sum(y_true, dim=-2, keepdim=True)  # (B, TO, N, 1)
        # y_out_true = torch.sum(y_true.permute(0, 1, 3, 2, 4), dim=-2, keepdim=True)  # (B, TO, N, 1)
        # y_pred, y_in, y_out = self.predict(batch)

        y_pred = self.predict(batch)

        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        # y_in_true = self._scaler.inverse_transform(y_in_true[..., :self.output_dim])
        # y_out_true = self._scaler.inverse_transform(y_out_true[..., :self.output_dim])

        y_pred = self._scaler.inverse_transform(y_pred[..., :self.output_dim])
        # y_in = self._scaler.inverse_transform(y_in[..., :self.output_dim])
        # y_out = self._scaler.inverse_transform(y_out[..., :self.output_dim])

        loss_pred = loss.masked_mse_torch(y_pred, y_true)
        # loss_in = loss.masked_mse_torch(y_in, y_in_true)
        # loss_out = loss.masked_mse_torch(y_out, y_out_true)
        # return self.loss_p0 * loss_pred + self.loss_p1 * loss_in + self.loss_p2 * loss_out
        return loss_pred


    def predict(self, batch):
        x = batch['X']  # (B, T, N, N, 1)
        # self.semantic_adj = generate_semantic_adj(x.squeeze(dim=-1), self.device)
        assert x.shape[-1] == 1 or print("The feature_dim must be 1")
        y_pred = []
        y_in_pred = []
        y_out_pred = []
        x_ = x.clone()
        for i in range(self.output_window):
            batch_tmp = {'X': x_}
            # y_, y_in_, y_out_ = self.forward(batch_tmp)  # (B, 1, N, N, 1)
            y_ = self.forward(batch_tmp)
            y_pred.append(y_.clone())
            # y_in_pred.append(y_in_.clone())
            # y_out_pred.append(y_out_.clone())

            x_ = torch.cat([x_[:, 1:, :, :, :], y_], dim=1)

        y_pred = torch.cat(y_pred, dim=1)  # (B, TO, N, N, 1)
        # y_in_pred = torch.cat(y_in_pred, dim=1)  # (B, TO, N, 1)
        # y_out_pred = torch.cat(y_out_pred, dim=1)  # (B, TO, N, 1)
        # return y_pred, y_in_pred, y_out_pred
        return y_pred
