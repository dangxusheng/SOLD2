import torch.nn as nn


class PixelShuffleDecoder(nn.Module):
    """ Pixel shuffle decoder. """

    def __init__(self, input_feat_dim=128, num_upsample=2, output_channel=2):
        super(PixelShuffleDecoder, self).__init__()
        # Get channel parameters
        self.channel_conf = self.get_channel_conf(num_upsample)

        # TODO： 通道裁剪
        ch_scale = 2
        # Define the pixel shuffle
        self.pixshuffle = nn.PixelShuffle(2)

        # Process the feature
        self.conv_block_lst = []
        # The input block
        self.conv_block_lst.append(
            nn.Sequential(
                nn.Conv2d(input_feat_dim // ch_scale, self.channel_conf[0] // ch_scale,
                          kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(self.channel_conf[0] // ch_scale),
                nn.ReLU(inplace=True)
            ))

        # Intermediate block
        for channel in self.channel_conf[1:-1]:
            self.conv_block_lst.append(
                nn.Sequential(
                    nn.Conv2d(channel // ch_scale, channel // ch_scale, kernel_size=3,
                              stride=1, padding=1),
                    nn.BatchNorm2d(channel // ch_scale),
                    nn.ReLU(inplace=True)
                ))

        # Output block
        self.conv_block_lst.append(
            nn.Conv2d(self.channel_conf[-1] // ch_scale, output_channel,
                      kernel_size=1, stride=1, padding=0)
        )
        self.conv_block_lst = nn.ModuleList(self.conv_block_lst)

    # Get num of channels based on number of upsampling.
    def get_channel_conf(self, num_upsample):
        if num_upsample == 2:
            return [256, 64, 16]
        elif num_upsample == 3:
            return [256, 64, 16, 4]

    def forward(self, input_features):
        # Iterate til output block
        out = input_features
        for block in self.conv_block_lst[:-1]:
            out = block(out)
            out = self.pixshuffle(out)

        # Output layer
        out = self.conv_block_lst[-1](out)

        return out
