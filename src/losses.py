import math
import torch
import torch.nn.functional as F


class SSIM(torch.nn.Module):
    """
    Calculate the Structural Similarity Index between two images.
    """

    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()

        self.window_size = window_size
        self.size_average = size_average

        self.window = None
        self.channels = None

    def forward(self, first_image, second_image):
        if len(first_image.size()) == 5:
            (batch, frames, channels, _, _) = first_image.size()
        else:
            (_, channels, _, _) = first_image.size()

        window = self.create_window(self.window_size, channels)
        window = window.type_as(first_image)

        if len(first_image.size()) == 5:
            mean = []
            for frame in range(frames):
                mean.append(
                    self.ssim(
                        first_image[:, frame, :, :, :],
                        second_image[:, frame, :, :, :],
                        window,
                        self.window_size,
                        channels,
                        self.size_average,
                    )
                )
            mean = torch.stack(mean, dim=0)

            return torch.mean(mean)
        else:
            return self.ssim(first_image, second_image, window, self.window_size, channels, self.size_average)

    def ssim(self, first_image, second_image, window, window_size, channels, size_average=True):
        mu1 = F.conv2d(first_image, window, padding=window_size // 2, groups=channels)
        mu2 = F.conv2d(second_image, window, padding=window_size // 2, groups=channels)

        mu1_squared = torch.pow(mu1, 2)
        mu2_squared = torch.pow(mu2, 2)
        mu1_mu2 = mu1 * mu2

        sigma1_squared = (
            F.conv2d(first_image * first_image, window, padding=window_size // 2, groups=channels) - mu1_squared
        )
        sigma2_squared = (
            F.conv2d(second_image * second_image, window, padding=window_size // 2, groups=channels) - mu2_squared
        )
        sigma12 = F.conv2d(first_image * second_image, window, padding=window_size // 2, groups=channels) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_squared + mu2_squared + C1) * (sigma1_squared + sigma2_squared + C2)
        )

        if size_average:
            return torch.mean(ssim_map)
        else:
            return torch.mean(torch.mean(torch.mean(ssim_map, 1), 1), 1)

    def create_window(self, window_size, channels):
        window_1D = self.gaussian(window_size, 1.5).unsqueeze(1)
        window_2D = window_1D.mm(window_1D.t()).float().unsqueeze(0).unsqueeze(0)
        window = window_2D.expand(channels, 1, window_size, window_size).contiguous()

        return window

    def gaussian(self, window_size, sigma):
        window = torch.Tensor(
            [math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2)) for x in range(window_size)]
        )

        return window / window.sum()


class L1_L2_Loss(torch.nn.Module):
    """
    Calculate the L1-L2 loss between two images.
    """

    def __init__(self):
        super(L1_L2_Loss, self).__init__()

    def forward(self, target, pred):
        error = target - pred
        loss = torch.abs(error) + torch.pow(error, 2)

        return torch.mean(loss)
