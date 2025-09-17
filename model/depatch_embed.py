import torch
import torch.nn as nn
from timm.models.layers import to_2tuple
from ms_deform_attn_func import MSDeformAttnFunction
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, patch_count=14, in_chans=3, embed_dim=768, with_norm=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_count = to_2tuple(patch_count)
        patch_stride_h = img_size[0] // patch_count[0]
        patch_stride_w = img_size[1] // patch_count[1]

        patch_pad_h = max(0, (patch_stride_h * (patch_count[0] - 1) + patch_size - img_size[0]) // 2)
        patch_pad_w = max(0, (patch_stride_w * (patch_count[1] - 1) + patch_size - img_size[1]) // 2)

        patch_size = to_2tuple(patch_size)
        num_patches = patch_count[0] * patch_count[1]
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.patch_count = patch_count

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=(patch_stride_h, patch_stride_w),
                              padding=(patch_pad_h, patch_pad_w))
        if with_norm:
            self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        expected_H, expected_W = self.img_size
        if H != expected_H or W != expected_W:
            patch_count_h, patch_count_w = self.patch_count
            if H % patch_count_h != 0 or W % patch_count_w != 0:
                raise ValueError(
                    f"Input image size ({H}*{W}) is not divisible by patch count "
                    f"({patch_count_h}*{patch_count_w})."
                )
            self.img_size = (H, W)

        x = self.proj(x).flatten(2).transpose(1, 2)
        if hasattr(self, "norm"):
            x = self.norm(x)
        actual_num_patches = x.shape[1]
        if actual_num_patches != self.num_patches:
            self.num_patches = actual_num_patches
        return x
class Simple_Patch(nn.Module):
    def __init__(self, offset_embed, img_size=224, patch_size=16, patch_pixel=16, patch_count=14,
                 in_chans=3, embed_dim=192, another_linear=False, use_GE=False, local_feature=False, with_norm=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_count = to_2tuple(patch_count)
        self.num_patches = patch_count[0] * patch_count[1]
        self.another_linear = another_linear
        if self.another_linear:
            self.patch_embed = PatchEmbed(img_size, 1 if local_feature else patch_size, patch_count, in_chans,
                                          embed_dim, with_norm=with_norm)
            self.act = nn.GELU() if use_GE else nn.Identity()
            self.offset_predictor = nn.Linear(embed_dim, offset_embed, bias=False)
        else:
            self.patch_embed = PatchEmbed(img_size, 1 if local_feature else patch_size, patch_count, in_chans,
                                          offset_embed)

        self.img_size, self.patch_size, self.patch_pixel, self.patch_count = img_size, patch_size, patch_pixel, patch_count
        self.in_chans, self.embed_dim = in_chans, embed_dim
    def reset_offset(self):
        if self.another_linear:
            nn.init.constant_(self.offset_predictor.weight, 0)
            if hasattr(self.offset_predictor, "bias") and self.offset_predictor.bias is not None:
                nn.init.constant_(self.offset_predictor.bias, 0)
        else:
            nn.init.constant_(self.patch_embed.proj.weight, 0)
            if hasattr(self.patch_embed.proj, "bias") and self.patch_embed.proj.bias is not None:
                nn.init.constant_(self.patch_embed.proj.bias, 0)
        print("Parameter for offsets reseted.")
    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x, model_offset=None):
        if x.dim() == 3:
            B, H, W = x.shape[0], self.img_size[0], self.img_size[1]
            assert x.shape[1] == H * W
            x = x.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        B, C, H, W = x.shape
        expected_H, expected_W = self.img_size
        if H != expected_H or W != expected_W:
            patch_count_h, patch_count_w = self.patch_count
            if H % patch_count_h != 0 or W % patch_count_w != 0:
                raise ValueError(
                    f"Input image size ({H}*{W}) is not divisible by patch count "
                    f"({patch_count_h}*{patch_count_w})."
                )
            self.img_size = (H, W)
            self.patch_embed.img_size = (H, W)
        img = x
        x = self.patch_embed(x)
        if self.another_linear:
            pred_offset = self.offset_predictor(self.act(x))
        else:
            pred_offset = x.contiguous()
        return self.get_output(img, pred_offset, model_offset), self.patch_count
class Simple_DePatch(Simple_Patch):
    def __init__(self, box_coder, show_dim=4, **kwargs):
        super().__init__(show_dim, **kwargs)
        self.box_coder = box_coder
        self.register_buffer("value_spatial_shapes",
                             torch.as_tensor([[self.img_size[0], self.img_size[1]]], dtype=torch.long))
        self.register_buffer("value_level_start_index", torch.as_tensor([0], dtype=torch.long))
        self.output_proj = nn.Linear(self.in_chans * self.patch_pixel * self.patch_pixel, self.embed_dim)
        self.num_sample_points = self.patch_pixel * self.patch_pixel * self.patch_count[0] * self.patch_count[1]
        if kwargs["with_norm"]:
            self.with_norm = True
            self.norm = nn.LayerNorm(self.embed_dim)
        else:
            self.with_norm = False

    def _init_buffers(self, img_size, device):
        # 如果尺寸相同，则跳过
        if torch.all(self.value_spatial_shapes == torch.tensor([[img_size[0], img_size[1]]], dtype=torch.long,device=device)):
            return
        self.value_spatial_shapes = torch.as_tensor([[img_size[0], img_size[1]]],
                                                    dtype=torch.long, device=device)
        self.value_level_start_index = torch.as_tensor([0], dtype=torch.long, device=device)
        self.num_sample_points = self.patch_pixel * self.patch_pixel * self.patch_count[0] * self.patch_count[1]

    def get_output(self, img, pred_offset, model_offset=None):
        B = img.shape[0]
        H, W = img.shape[2], img.shape[3]
        device = img.device
        self._init_buffers((H, W), device)
        sample_location = self.box_coder(pred_offset, model_offset)
        if sample_location.dim() == 3 and sample_location.shape[2] == 2:
            num_patches = sample_location.shape[1]
            sample_location = sample_location.unsqueeze(2).unsqueeze(2).unsqueeze(2)
            sample_location = sample_location.repeat(1, 1, self.patch_pixel, self.patch_pixel, 1, 1)
            sample_location = sample_location.reshape(B, -1, 1, 1, 1, 2)
        elif sample_location.dim() == 3 and sample_location.shape[2] == 4:
            num_patches = sample_location.shape[1]
            centers = (sample_location[:, :, :2] + sample_location[:, :, 2:]) / 2
            centers = centers.unsqueeze(2).unsqueeze(2).unsqueeze(2)
            centers = centers.repeat(1, 1, self.patch_pixel, self.patch_pixel, 1, 1)
            sample_location = centers.reshape(B, -1, 1, 1, 1, 2)
        else:
            raise ValueError(f"Unexpected sample_location shape: {sample_location.shape}")
        attention_weights = torch.ones((B, self.num_sample_points, 1, 1, 1), device=device)
        x = img.reshape(B, self.in_chans, 1, -1).transpose(1, 3).contiguous()
        output = MSDeformAttnFunction.apply(x, self.value_spatial_shapes, self.value_level_start_index, sample_location,
                                            attention_weights, 1)
        output = output.reshape(B, self.num_patches, self.in_chans * self.patch_pixel * self.patch_pixel)
        output = self.output_proj(output)
        if self.with_norm:
            output = self.norm(output)
        return output