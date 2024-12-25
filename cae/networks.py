from torch import nn
import torch
import torch.nn.functional as F

criterion = nn.CrossEntropyLoss()

class MultiClassDis(nn.Module):
    def __init__(self, device):
        super(MultiClassDis, self).__init__()
        self.device = device

        self.n_layer = 4
        self.dim = 64
        self.norm = 'gn'
        self.activ = 'silu'

        self.input_dim = 3
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.cnns = self._make_net()
        self.num_classes = 6
        self.fc = nn.Sequential(
            nn.Linear(self.dim * (pow(2, (self.n_layer - 1))), 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, self.num_classes)
        )

    def _make_net(self):
        dim = self.dim

        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ)]
        for i in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ)]
            dim *= 2
        cnn_x += [ResBlocks(2, dim, norm=self.norm, activation=self.activ)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):

        output=self.fc(torch.flatten(self.avgpool(self.cnns(x)), 1))
        return output

    def calc_dis_loss(self, input_fake=None, input_real=None, real_label=None):
        # calculate the loss to train D

        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        # real_label = real_label.squeeze(1)
        dis_loss = criterion(outs0[:, :2], torch.zeros(outs0.size(0)).long().to(
            self.device)) + criterion(outs1[:, :2],torch.ones(outs1.size(0)).long().to(self.device))

        cla_loss = criterion(outs1[:, 2:], real_label)

        dis_cla_loss_total = 1*dis_loss +  2*cla_loss

        # if outs1[:, 2:].argmax(dim=-1) == real_label: # 如果real图像分类正确，则classify_right返回1，否则返回0
        #     classify_right = 1
        # else:
        #     classify_right = 0
        # print(outs1[:, 2:], outs1[:, 2:].argmax(dim=-1), real_label, classify_right)

        # return dis_cla_loss_total       #, classify_right
        return dis_cla_loss_total, dis_loss, cla_loss


    def calc_gen_loss(self, input_fake=None, real_label=None):
        # calculate the loss to train G

        outs0 = self.forward(input_fake)
        # real_label = real_label.squeeze(1)
        dis_loss = criterion(outs0[:, :2], torch.ones(outs0.size(0)).long().to(self.device))
        cla_loss = criterion(outs0[:, 2:], real_label)
        dis_cla_loss_total = dis_loss + cla_loss

        # if outs0[:, 2:].argmax(dim=-1) == real_label: # 如果fake图像分类正确，则exchange_right返回1，说明类别切换成功，否则返回0
        #     exchange_right = 1
        # else:
        #     exchange_right = 0
        # print(outs0[:, 2:], outs0[:, 2:].argmax(dim=-1), real_label, exchange_right)

        # return dis_cla_loss_total, exchange_right
        return dis_cla_loss_total, dis_loss, cla_loss



class CAEGen(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self):
        super(CAEGen, self).__init__()

        # style encoder
        self.enc_style = StyleEncoder(4, 3, 64, 512, norm='none', activ='silu')

        # content encoder
        self.enc_content = ContentEncoder(2, 6, 3, 64, 'gn', activ='silu')

        self.dec = Decoder(2, 6, self.enc_content.output_dim, 3, activ='silu')

    def forward(self, images):
        # reconstruct an image
        content, style_fake = self.encode(images)
        images_recon = self.decode(content, style_fake)
        return images_recon

    def encode(self, images):
        # encode an image to its content and style codes
        style_fake = self.enc_style(images)
        content = self.enc_content(images)
        return content, style_fake

    def decode(self, content, style):
        images = self.dec(content, style)
        return images



class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ):
        super(StyleEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ)]

        self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.model += [nn.AdaptiveAvgPool2d(1)]
        self.model = nn.Sequential(*self.model)

        self.linear = LinearBlock(style_dim, style_dim, norm='none', activation='none')

    def forward(self, x):
        x = self.model(x)
        x = self.linear(x.view(x.size(0), -1))
        return x


class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ)]
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ)]
            dim *= 2
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, activ='relu'):
        super(Decoder, self).__init__()

        self.model = []
        self.emb_res_blocks = EmbResBlocks(n_res, dim, activ)

        # upsampling blocks
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='gn', activation=activ)]
            dim //= 2
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='none')]

        self.model = nn.Sequential(*self.model)

    def forward(self, x, emb):
        x = self.emb_res_blocks(x, emb)
        return self.model(x)


class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class EmbResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, activation='relu'):
        super(EmbResBlocks, self).__init__()
        self.model = nn.ModuleList([])
        for i in range(num_blocks):
            self.model.append(EmbResBlock(dim, activation=activation))

    def forward(self, x, emb):
        for layer in self.model:
            x = layer(x, emb)
        return x


class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation)]
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation='none')]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class EmbResBlock(nn.Module):
    def __init__(self, dim, activation='relu'):
        super(EmbResBlock, self).__init__()

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        if activation == 'silu':
            self.activation = nn.SiLU(inplace=True)

        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)

        self.mlp1 = LinearBlock(512, dim * 2, norm='none', activation='none')
        self.mlp2 = LinearBlock(512, dim * 2, norm='none', activation='none')

    def forward(self, x, emb):
        residual = x
        x = self.conv1(x)

        emb1 = self.mlp1(emb)
        while len(emb1.shape) < len(x.shape):
            emb1 = emb1[..., None]

        shift1, scale1 = emb1.chunk(2, dim=1)
        x = x * shift1 + scale1

        x = self.activation(x)
        x = self.conv2(x)

        emb2 = self.mlp2(emb)
        while len(emb2.shape) < len(x.shape):
            emb2 = emb2[..., None]

        shift2, scale2 = emb2.chunk(2, dim=1)
        x = x * shift2 + scale2

        out = x + residual

        return out




class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='silu'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = nn.LayerNorm(norm_dim)
        elif norm == 'gn':
            self.norm = nn.GroupNorm(32, norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding, bias=True)

    def forward(self, x):
        x = self.conv(x)

        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = nn.LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


