import torch
import torch.nn as nn

# from fast_transformers.builders import TransformerEncoderBuilder
import time
from .linear_transformer import LinearTransformerBlock


class JiafengCNN(nn.Module):
    def __init__(self, input_channels, target_sources_num, depth=6, use_trans=False):
        super(JiafengCNN, self).__init__()

        in_channels = 2
        channels = 64

        num_heads = 32
        kernel_size = 8192
        seq_len = 524288 // 2  # 262144

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for index in range(depth):
            seq_len = int(seq_len // 4)

            # ------ encoder part ---------
            encode = []
            encode += [
                nn.Conv1d(in_channels, channels, kernel_size=8, stride=4, padding=2),
                nn.ReLU(),
            ]

            if use_trans:
                # adjust transformer input length for each layer
                if seq_len > 2048:
                    encode += [
                        TransformerLayer(
                            channels=channels, num_heads=num_heads, seq_len=kernel_size
                        ),
                        nn.ReLU(),
                    ]
                else:
                    encode += [
                        TransformerLayer(
                            channels=channels, num_heads=num_heads, seq_len=seq_len
                        ),
                        nn.ReLU(),
                    ]
            else:
                encode += [
                    nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                ]

            self.encoder.append(nn.Sequential(*encode))
            kernel_size = int(kernel_size // 2)

            # ------ decoder part ---------
            decode = []
            if index > 0:
                out_channels = in_channels
            else:
                out_channels = 2

            if use_trans:
                if seq_len > 2048:
                    decode += [
                        TransformerLayer(
                            channels=channels, num_heads=num_heads, seq_len=kernel_size
                        ),
                        nn.ReLU(),
                    ]
                else:
                    decode += [
                        TransformerLayer(
                            channels=channels, num_heads=num_heads, seq_len=seq_len
                        ),
                        nn.ReLU(),
                    ]
            else:
                decode += [
                    nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                ]

            decode += [
                nn.ConvTranspose1d(
                    channels, out_channels, kernel_size=8, stride=4, padding=2
                )
            ]

            if index > 0:
                decode.append(nn.ReLU())

            self.decoder.insert(0, nn.Sequential(*decode))

            in_channels = channels
            channels = int(2 * channels)

    def forward(
        self,
        input_dict,
        audio_input=None,
        audio_target=None,
    ):

        # input shape: (1, 2, 442368)
        audio_input = input_dict['waveform']
        x = audio_input
        # x = F.pad(audio_input, (0, 1368), "constant", 0)
        # x = F.pad(audio_input, (0, 344), "constant", 0)

        inf_start = time.time()
        saved = []
        for layer_idx, encode in enumerate(self.encoder):
            x = encode(x)
            saved.append(x)

        for layer_idx, decode in enumerate(self.decoder):
            skip = saved.pop(-1)
            x = x + skip
            x = decode(x)
        # print("Inference: ", time.time() - inf_start)

        # logits = x[:, :, :441000]
        logits = x

        # output_dict = {'wav': logits}
        output_dict = {'waveform': logits}

        return output_dict


class JiafengTTNet(nn.Module):
    def __init__(self, input_channels, target_sources_num, depth=6, use_trans=True):
        super(JiafengTTNet, self).__init__()

        in_channels = 2
        channels = 64

        num_heads = 32
        kernel_size = 8192
        seq_len = 524288 // 2  # 262144

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for index in range(depth):
            seq_len = int(seq_len // 4)

            # ------ encoder part ---------
            encode = []
            encode += [
                nn.Conv1d(in_channels, channels, kernel_size=8, stride=4, padding=2),
                nn.ReLU(),
            ]

            if use_trans:
                # adjust transformer input length for each layer
                if seq_len > 2048:
                    encode += [
                        TransformerLayer(
                            channels=channels, num_heads=num_heads, seq_len=kernel_size
                        ),
                        nn.ReLU(),
                    ]
                else:
                    encode += [
                        TransformerLayer(
                            channels=channels, num_heads=num_heads, seq_len=seq_len
                        ),
                        nn.ReLU(),
                    ]
            else:
                encode += [
                    nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                ]

            self.encoder.append(nn.Sequential(*encode))
            kernel_size = int(kernel_size // 2)

            # ------ decoder part ---------
            decode = []
            if index > 0:
                out_channels = in_channels
            else:
                out_channels = 2

            if use_trans:
                if seq_len > 2048:
                    decode += [
                        TransformerLayer(
                            channels=channels, num_heads=num_heads, seq_len=kernel_size
                        ),
                        nn.ReLU(),
                    ]
                else:
                    decode += [
                        TransformerLayer(
                            channels=channels, num_heads=num_heads, seq_len=seq_len
                        ),
                        nn.ReLU(),
                    ]
            else:
                decode += [
                    nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                ]

            decode += [
                nn.ConvTranspose1d(
                    channels, out_channels, kernel_size=8, stride=4, padding=2
                )
            ]

            if index > 0:
                decode.append(nn.ReLU())

            self.decoder.insert(0, nn.Sequential(*decode))

            in_channels = channels
            channels = int(2 * channels)

    def forward(
        self,
        input_dict,
        audio_input=None,
        audio_target=None,
    ):

        # input shape: (1, 2, 442368)
        audio_input = input_dict['waveform']
        x = audio_input
        # x = F.pad(audio_input, (0, 1368), "constant", 0)
        # x = F.pad(audio_input, (0, 344), "constant", 0)

        inf_start = time.time()
        saved = []
        for layer_idx, encode in enumerate(self.encoder):
            x = encode(x)
            saved.append(x)

        for layer_idx, decode in enumerate(self.decoder):
            skip = saved.pop(-1)
            x = x + skip
            x = decode(x)
        # print("Inference: ", time.time() - inf_start)

        # logits = x[:, :, :441000]
        logits = x

        # output_dict = {'wav': logits}
        output_dict = {'waveform': logits}

        return output_dict


class TransformerLayer(nn.Module):
    def __init__(
        self,
        channels,
        num_heads,
        seq_len,
        padding=0,
    ):
        super(TransformerLayer, self).__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.seq_len = seq_len

        self.padding = padding

        self.unfold = nn.Unfold(
            kernel_size=(self.seq_len, 1),
            stride=(self.seq_len, 1),
            padding=(self.padding, 0),
        )

        self.trans = LinearTransformerBlock(self.channels, self.num_heads)
        self.pos = nn.Parameter(torch.zeros(1, self.seq_len, self.channels))

    def forward(self, x):
        # ----------- unfold tensor to packs ----------------
        x = x.unsqueeze(-1)
        x = self.unfold(x)
        bs, _, num_packs = x.size()
        x = x.view(bs, self.channels, self.seq_len, num_packs).permute(0, 3, 2, 1)
        x = x.reshape(bs * num_packs, self.seq_len, self.channels)

        # ------------ go to transformers ---------------
        x = x + self.pos
        x = self.trans(x)

        # ----------- reshape back ---------------------
        x = x.reshape(bs, num_packs * self.seq_len, self.channels)
        x = x.transpose(1, 2)

        return x


#
# device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
# model = TTnet()
#
# pytorch_total_params = sum(p.numel() for p in model.parameters())
# print(pytorch_total_params)
#
# model.to(device)
# model.train()
#
# rand_tensor = torch.rand(1, 2, 524288 // 2)
# inputs = {
#     'audio_input': rand_tensor.to(device),
#     "audio_target": rand_tensor.to(device),
#           }
#
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
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
#
#             loss_start = time.time()
#             loss = torch.mean(torch.abs(outputs['wav'] - inputs['audio_target']))
#
#             loss_mid = time.time()
#             # print("loss mean", loss_mid - loss_start)
#
#             torch.cuda.empty_cache()
#             loss.backward()
#             # print("loss backward", time.time() - loss_mid)
#             optimizer.step()
#
#             print(epoch, it, " ", loss, lr_scheduler.get_lr())
