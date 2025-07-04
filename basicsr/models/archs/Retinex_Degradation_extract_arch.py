import torch
import torch.nn as nn
from basicsr.models.archs.Transformer import Transformer
from basicsr.models.archs.diffusion import diffusion
from basicsr.models.archs.Encoder import EfficientIlluminationDecoder, denoise
from basicsr.models.archs.Img_Restoration import LYT

class Retinex_Degradation_pretrainv2(nn.Module):
    def __init__(self):
        super(Retinex_Degradation_pretrainv2, self).__init__()
        timesteps = 4
        linear_start = 0.1
        linear_end = 0.99
        hidden_dim = 512
        self.LYT = LYT().cuda()
        self.encoder = EfficientIlluminationDecoder(in_channels=7, hidden_dim=512).cuda()
        self.condition = EfficientIlluminationDecoder(in_channels=4, hidden_dim=512).cuda()
        self.denoise = denoise(hidden_dim=512, feats=64, timesteps=4).cuda()
        self.diffusion = diffusion.DDPM(denoise=self.denoise,
                                        condition=self.condition, feats=64, timesteps=timesteps,
                                        linear_start=linear_start, linear_end=linear_end).cuda()
        self.Restorer = Transformer(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 4]).cuda()
            
    def _rgb_to_ycbcr(self, image):
        # image: (batch_size, 3, height, width) tensor, with values in range [0, 1]
        r, g, b = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.14713 * r - 0.28886 * g + 0.436 * b + 0.5
        v = 0.615 * r - 0.51499 * g - 0.10001 * b + 0.5
        yuv = torch.stack((y, u, v), dim=1)
        return yuv
        
    def _ycbcr_to_rgb(self, yuv):
        # yuv: (batch_size, 3, height, width) tensor, with values in range [0, 1]
        y, u, v = yuv[:, 0, :, :], yuv[:, 1, :, :], yuv[:, 2, :, :]

        # Inverse transformation (YCbCr to RGB)
        u = u - 0.5  # shift U channel
        v = v - 0.5  # shift V channel

        r = y + 1.13983 * v
        g = y - 0.39465 * u - 0.58060 * v
        b = y + 2.03211 * u
        rgb = torch.stack((r, g, b), dim=1)
        return rgb


    def forward(self, x):
        img = x[0]
        gt = x[1]
        b, *_ = img.shape

        process_img = self.LYT(img)
        img = self._rgb_to_ycbcr(img)
        gt = self._rgb_to_ycbcr(gt)

        combine_gt_img = torch.cat((img, process_img, gt), dim=1)
        cdp = self.encoder(combine_gt_img)  # cdp:B*3*H*W
        cdp_diff = cdp
        image_restored = self.Restorer(img, process_img, cdp_diff)
        # image_restored = image_restored = torch.clamp(image_restored, 0.0, 1.0)  # 保证数值在[0, 1]范围内\
        # Can compress dynamic range of values/ensure smooth gradients during training
        image_restored = torch.sigmoid(image_restored)
        enhancement_image = self._ycbcr_to_rgb(image_restored)
        if self.training:
            # 返回训练阶段所需的多种输出
            return {
                'enhancement_image': enhancement_image,
                "process_img": process_img,
                "process_gt_img": gt[:, 0, :, :].unsqueeze(1),
            }
        else:
            return enhancement_image

if __name__ == '__main__':
    import torch
    from thop import profile

    def count_module_flops(module, input_size, device):
        """
        Count FLOPs for a specific module
        """
        module = module.to(device)
        x = torch.randn(input_size).to(device)
        flops, _ = profile(module, inputs=(x,), verbose=False)
        return flops
        
    model = Retinex_Degradation_pretrainv2()
    
    # 统计每个模块的参数量
    module_params = {
        'LYT': sum(p.nelement() for p in model.LYT.parameters()),
        'encoder': sum(p.nelement() for p in model.encoder.parameters()),
        'condition': sum(p.nelement() for p in model.condition.parameters()),
        'denoise': sum(p.nelement() for p in model.denoise.parameters()),
        'Restorer': sum(p.nelement() for p in model.Restorer.parameters())
    }
    
    total_params = sum(module_params.values())
    
    # 打印每个模块的参数量和百分比
    print('模块参数统计:')
    print('-' * 50)
    print(f"{'模块名':<15} {'参数量(M)':<12} {'占比(%)':<10}")
    print('-' * 50)
    
    for module_name, params in module_params.items():
        percentage = (params / total_params) * 100
        params_m = params / (1000 * 1000)
        print(f"{module_name:<15} {params_m:>8.2f}M    {percentage:>6.2f}%")
    
    print('-' * 50)
    print(f"总参数量: {total_params/(1000*1000):.2f}M")
    
    # 计算FLOPs
    flops = count_module_flops(model, (2, 1, 3, 256, 256), device=torch.device('cuda'))
    print(f'FLOPs: {flops / (1024*1024*1024):.2f}G')