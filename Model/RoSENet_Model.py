import torch
from torch import nn
from torch import cosine_similarity
class SASSC(nn.Module):
    def __init__(self, in_channels_fused, out_channels):
        super(SASSC, self).__init__()
        self.ED_Cos = ED_Cos()
        self.CSS_C_1 = CSS_Conv(in_channels_fused, out_channels, 1)
        self.CSS_C_2 = CSS_Conv(out_channels, out_channels, 3)
    def forward(self, x_fused):
        x = self.ED_Cos(x_fused)
        x = self.CSS_C_1(x)
        x = self.CSS_C_2(x)
        return x
class ESSCA(nn.Module):
    def __init__(self, in_channels):
        super(ESSCA, self).__init__()
        self.ED = ED(in_channels)
        self.bn_2d = nn.BatchNorm2d(in_channels)
        self.scAttention = scAttention(in_channels)
    def forward(self, x):
        x = self.ED(x)
        x = self.bn_2d(x)
        x = self.scAttention(x)
        return x

class mlp(nn.Module):
    def __init__(self, dim, mlp_dim,class_count):
        super().__init__()
        self.fclayer1 = nn.Linear(dim, mlp_dim)
        self.fclayer2 = nn.Linear(mlp_dim, class_count)
        self.act_fn = nn.Softmax()
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        x = self.fclayer1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fclayer2(x)
        x = self.dropout(x)
        return x

class ED_Cos(nn.Module):
    def __init__(self):
        super(ED_Cos, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.lamuda = nn.Parameter(torch.tensor(0.5, dtype=float), requires_grad=True)
    def forward(self, x):
        batch_size, c, h, w = x.size()
        q, k = x.view(batch_size, -1, h * w).permute(0, 2, 1), x.view(batch_size, -1, h * w)
        cent_spec_vector = q[:, int((h * w - 1) / 2)]
        cent_spec_vector = torch.unsqueeze(cent_spec_vector, 1)
        csv_expand = cent_spec_vector.expand(batch_size, h * w, c)
        E_dist = torch.norm(csv_expand - q, dim=2, p=2)
        sim_E_dist = 1 / (1 + E_dist)
        sim_cos = cosine_similarity(cent_spec_vector, k.permute(0, 2, 1), dim=2)  # include negative
        lmd = torch.sigmoid(self.lamuda)
        atten_ED = self.softmax(sim_E_dist)
        atten_cos = self.softmax(sim_cos)
        atten_s = lmd * atten_ED + (1 - lmd) * atten_cos
        atten_s = torch.unsqueeze(atten_s, 2)
        q_attened = torch.mul(atten_s, q)
        out = q_attened.contiguous().view(batch_size, -1, h, w) + x
        return out


class ED(nn.Module):
    def __init__(self, in_channels):
        super(ED, self).__init__()
        self.to_qkv = nn.Conv2d(in_channels, in_channels * 3, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, fmap):
        batch_size, c, h, w = fmap.size()
        q, k, v = self.to_qkv(fmap).view(batch_size, -1, h * w).permute(0, 2, 1).chunk(3, dim=-1)
        cent_spec_vector = q[:, int((h * w - 1) / 2)]
        cent_spec_vector = torch.unsqueeze(cent_spec_vector, 1)
        csv_expand = cent_spec_vector.expand(batch_size, h * w, c)
        E_dist = torch.norm(csv_expand - k, dim=2, p=2)
        sim_E_dist = 1 / (1 + E_dist)
        atten_ED = self.softmax(sim_E_dist)
        atten_sim = torch.unsqueeze(atten_ED, 2)
        v_attened = torch.mul(atten_sim, v)
        out = v_attened.contiguous().view(batch_size, -1, h, w) + fmap
        return out

class CSS_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(CSS_Conv, self).__init__()
        self.point_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.depth_conv = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=out_channels,
        )
        self.leaky = nn.LeakyReLU(inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.BN = nn.BatchNorm2d(in_channels)
    def forward(self, input):
        out = self.point_conv(self.BN(input))
        out = self.leaky(out)
        out = self.depth_conv(out)
        out = self.relu(out)
        return out


class RoseNet_body(nn.Module):
    def __init__(self, in_channels, in_channels_fused, class_count):
        super(RoseNet_body, self).__init__()
        self.class_count = class_count
        self.in_channels = in_channels
        self.in_channels_fused = in_channels_fused
        self.out_channels = in_channels_fused
        self.relu = nn.ReLU(inplace=True)
        self.SASSC = SASSC(self.in_channels_fused,self.out_channels)
        self.ESSCA = ESSCA(self.out_channels)
    def forward(self, x1, x2):
        x_fused = torch.cat((x1, x2), dim=1)
        x = self.SASSC(x_fused)
        x = self.ESSCA(x)
        out = x
        return out

class scAttention(nn.Module):
    def __init__(self, in_channels, rate=4):
        super(scAttention, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels / rate), in_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(in_channels)
        )
    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2).sigmoid()
        x = x * x_channel_att
        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att
        return out

class RoseNet(nn.Module):
    def __init__(self, in_channels,in_channels_fused, h, w, class_count):
        super(RoseNet, self).__init__()
        self.class_count = class_count
        self.in_channels = in_channels
        self.in_channels_fused = in_channels_fused
        self.height_p = h
        self.width_p = w
        self.win_spa_size = self.height_p * self.width_p
        self.fc = nn.Linear(self.in_channels_fused, self.class_count)
        self.RoseNet_body = RoseNet_body(self.in_channels, self.in_channels_fused, self.class_count)
    def forward(self, x1, x2):
        x = self.RoseNet_body(x1,x2)
        x = x.mean(dim=(2, 3))
        out = self.fc(x)
        return out


