import torch
import torch.nn as nn
from fast_transformers.builders import TransformerEncoderBuilder
import torch.nn.functional as F


class TTnetNoTransformer(nn.Module):
    def __init__(self,
        input_channels,
        target_sources_num,
                 depth=6,
                 ):
        super(TTnetNoTransformer, self).__init__()

        in_channels = input_channels
        channels = 64
        kernel_size = 4096

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        seq_len = 442368
        for index in range(depth):
            seq_len = int(seq_len // 4)
            encode = []
            # --------------------------------
            # encode += [nn.Conv1d(in_channels, channels, kernel_size=8, stride=4, padding=2), nn.ReLU()]
            # encode += [nn.Conv1d(channels, 2 * channels, 1), nn.GLU(dim=1)]
            # self.encoder.append(nn.Sequential(*encode))
            # ---------------------------------

            if index < 3:
                encode += [TransformerEncoderLayer(in_channel=in_channels, out_channel=channels, kernel_size=kernel_size, stride=kernel_size, padding=0), nn.ReLU()]
                self.encoder.append(nn.Sequential(*encode))
                kernel_size = int(kernel_size // 4)
            else:
                encode += [TransformerEncoderLayer(in_channel=in_channels, out_channel=channels, kernel_size=seq_len, stride=seq_len, padding=0), nn.ReLU()]
                self.encoder.append(nn.Sequential(*encode))

            decode = []
            if index > 0:
                out_channels = in_channels
            else:
                out_channels = input_channels * target_sources_num

            # -----------------------------
            # decode += [nn.Conv1d(channels, 2 * channels, kernel_size=3, stride=1, padding=1), nn.GLU(dim=1)]
            # decode += [nn.ConvTranspose1d(channels, out_channels, kernel_size=8, stride=4, padding=2)]
            # -----------------------------

            if index < 3:
                decode += [TransformerDecoderLayer(in_channel=channels, out_channel=out_channels, kernel_size=kernel_size, stride=kernel_size, padding=0)]
            else:
                decode += [TransformerDecoderLayer(in_channel=channels, out_channel=out_channels, kernel_size=seq_len, stride=seq_len, padding=0)]

            if index > 0:
                decode.append(nn.ReLU())

            self.decoder.insert(0, nn.Sequential(*decode))

            in_channels = channels
            channels = int(2 * channels)


    def forward(self,
        input_dict
                # audio_input=None,
                # audio_target=None,
                ):

        audio_input = input_dict['waveform']

        # input shape: (1, 2, 442368)
        x = F.pad(audio_input, (0, 1368), "constant", 0)
        # x = F.pad(audio_input, (0, 344), "constant", 0)

        saved = []
        for encode in self.encoder:
            x = encode(x)
            saved.append(x)

        for decode in self.decoder:
            skip = saved.pop(-1)
            x = x + skip
            x = decode(x)

        # center shape (bs, 256, 432)

        logits = x[:, :, :441000]

        # output_dict = {'wav': logits}
        output_dict = {'waveform': logits}

        return output_dict


class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 stride,
                 padding,
                 ):
        super(TransformerEncoderLayer, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # self.unfold = nn.Unfold(kernel_size=(self.kernel_size, 1), stride=(self.stride, 1), padding=(self.padding, 0), )

        # self.trans = TransformerEncoderBuilder.from_kwargs(
        #     n_layers=1,
        #     n_heads=32,
        #     query_dimensions=self.out_channel // 32,
        #     value_dimensions=self.out_channel // 32,
        #     feed_forward_dimensions=self.out_channel * 4,
        #     attention_type="linear").get()
        # self.pos = nn.Parameter(torch.zeros(1, self.kernel_size, self.out_channel))

        self.cnn = nn.Conv1d(self.in_channel, self.out_channel, kernel_size=8, stride=4, padding=2)

    def forward(self, x):
        in_channel = x.size()[1]
        x = self.cnn(x)
        
        # x = x.unsqueeze(-1)
        # x = self.unfold(x)
        # bs, _, num_packs = x.size()
        # x = x.view(bs, self.out_channel, self.kernel_size, num_packs).permute(0, 3, 2, 1)
        # x = x.reshape(bs * num_packs, self.kernel_size, self.out_channel)

        # # -------------
        # x = x + self.pos
        # x = self.trans(x)
        # # -------------
        # x = x.reshape(bs, num_packs * self.kernel_size, self.out_channel)
        # # x = F.max_pool2d(x, kernel_size=(self.kernel_size, 1)).squeeze(1)  # (110592, 16)
        # # x = x.view(bs, num_packs, self.out_channel, )  # (1, 110592, 16)
        # #
        # x = x.transpose(1, 2)  # (1, 16, 110592)


        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 stride,
                 padding,
                 ):
        super(TransformerDecoderLayer, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # self.unfold = nn.Unfold(kernel_size=(self.kernel_size, 1), stride=(self.stride, 1), padding=(self.padding, 0), )

        # self.trans = TransformerEncoderBuilder.from_kwargs(
        #     n_layers=1,
        #     n_heads=32,
        #     query_dimensions=self.in_channel // 32,
        #     value_dimensions=self.in_channel // 32,
        #     feed_forward_dimensions=self.in_channel * 4,
        #     attention_type="linear").get()
        # self.pos = nn.Parameter(torch.zeros(1, self.kernel_size, self.in_channel))

        self.t_cnn = nn.ConvTranspose1d(self.in_channel, self.out_channel, kernel_size=8, stride=4, padding=2)

    def forward(self, x):

        # in_channel = x.size()[1]
        # x = x.unsqueeze(-1)
        # x = self.unfold(x)
        # bs, _, num_packs = x.size()
        # x = x.view(bs, self.in_channel, self.kernel_size, num_packs).permute(0, 3, 2, 1)
        # x = x.reshape(bs * num_packs, self.kernel_size, self.in_channel)
        # # -------------
        # x = x + self.pos
        # x = self.trans(x)
        # # -------------
        # x = x.reshape(bs, num_packs * self.kernel_size, self.in_channel)
        # # x = F.max_pool2d(x, kernel_size=(self.kernel_size, 1)).squeeze(1)  # (110592, 16)
        # # x = x.view(bs, num_packs, self.out_channel, )  # (1, 110592, 16)
        # #
        # x = x.transpose(1, 2)  # (1, 16, 110592)
        # from IPython import embed; embed(using=False); os._exit(0)
        x = self.t_cnn(x)

        return x

#
# test_layer = TransformerDecoderLayer(in_channel=256, out_channel=128, kernel_size=8, stride=4, padding=2)
# test_layer(torch.rand(1, 256, 442))

# test_layer = TransformerEncoderLayer(in_channel=12, out_channel=256, kernel_size=256, stride=256, padding=0)
# test_layer(torch.rand(1, 128, 27520))


# device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
# model = TTnet()
#
# pytorch_total_params = sum(p.numel() for p in model.parameters())
# print(pytorch_total_params)
#
# model.to(device)
# model.train()
#
# rand_tensor = torch.rand(1, 2, 441000)
# inputs = {
#     'audio_input': rand_tensor.to(device),
#     "audio_target": rand_tensor.to(device),
#           }
#
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
# decayRate = 1
# lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
# # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
#
# for epoch in range(5000):
#     lr_scheduler.step()
#     for it in range(100):
#         with torch.set_grad_enabled(True):
#             model.zero_grad()
#             outputs = model(**inputs)
#             torch.cuda.empty_cache()
#             loss = torch.mean(torch.abs(outputs['wav'] - inputs['audio_target']))
#             torch.cuda.empty_cache()
#             loss.backward()
#             optimizer.step()
#
#             print(epoch, it, " ", loss, lr_scheduler.get_lr())
