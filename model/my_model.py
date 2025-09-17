import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os
os.environ['TORCH_HOME'] = '../../pretrained_models'
import timm
from timm.models.layers import to_2tuple, trunc_normal_
from box_coder import pointCoder
from depatch_embed import Simple_DePatch
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
class ASTModel_Deformable(nn.Module):
    def __init__(self, label_dim=2, input_freq=224, input_time=224,
                 imagenet_pretrain=False, audioset_pretrain=False, model_size='tiny224',
                 use_depatch=True, verbose=True):
        super(ASTModel_Deformable, self).__init__()

        self.use_depatch = use_depatch
        self.verbose = verbose
        self.input_freq = input_freq
        self.input_time = input_time

        if verbose:
            print('---------------AST Model with DePatch Summary---------------')
            print('ImageNet pretraining: {:s}, AudioSet pretraining: {:s}'.format(
                str(imagenet_pretrain), str(audioset_pretrain)))
            print('Using DePatch: {:s}'.format(str(use_depatch)))
            print('Input size: {:d}x{:d}'.format(input_freq, input_time))
        if use_depatch:
            self.setup_depatch_model(label_dim, input_freq, input_time, imagenet_pretrain, audioset_pretrain,
                                     model_size)
        else:
            self.setup_standard_model(label_dim, input_freq, input_time, imagenet_pretrain, audioset_pretrain,
                                      model_size)

    def setup_depatch_model(self, label_dim, input_freq, input_time, imagenet_pretrain, audioset_pretrain, model_size):
        if model_size == 'tiny224':
            self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=imagenet_pretrain)
        elif model_size == 'small224':
            self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=imagenet_pretrain)
        elif model_size == 'base224':
            self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=imagenet_pretrain)
        elif model_size == 'base384':
            self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)
        else:
            raise Exception('Model size must be one of tiny224, small224, base224, base384.')

        self.original_embedding_dim = self.v.pos_embed.shape[2]
        self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim),
                                      nn.Linear(self.original_embedding_dim, label_dim))
        patch_size = 16
        patch_count_freq = input_freq // patch_size
        patch_count_time = input_time // patch_size
        num_patches = patch_count_freq * patch_count_time
        if self.verbose:
            print('patch size={:d}, patch count={:d}x{:d}'.format(patch_size, patch_count_freq, patch_count_time))
        box_coder = pointCoder(input_size=(input_freq, input_time), patch_count=(patch_count_freq, patch_count_time))
        self.v.patch_embed = Simple_DePatch(
            box_coder=box_coder,
            img_size=(input_time, input_freq),
            patch_size=patch_size,
            patch_pixel=8,
            patch_count=(patch_count_freq, patch_count_time),
            in_chans=1,
            embed_dim=self.original_embedding_dim,
            another_linear=True,
            use_GE=False,
            local_feature=False,
            with_norm=False
        )
        if next(self.parameters()).is_cuda:
            device = next(self.parameters()).device
            self.v.patch_embed.box_coder = self.v.patch_embed.box_coder.to(device)
        self.adjust_positional_embedding(num_patches, imagenet_pretrain, (patch_count_freq, patch_count_time))
    def setup_standard_model(self, label_dim, input_freq, input_time, imagenet_pretrain, audioset_pretrain, model_size):

        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        if audioset_pretrain == False:
            if model_size == 'tiny224':
                self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'small224':
                self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base224':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base384':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)
            else:
                raise Exception('Model size must be one of tiny224, small224, base224, base384.')
            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim),
                                          nn.Linear(self.original_embedding_dim, label_dim))

            patch_size = 16
            patch_count_freq = input_freq // patch_size
            patch_count_time = input_time // patch_size
            num_patches = patch_count_freq * patch_count_time
            self.v.patch_embed.num_patches = num_patches
            if self.verbose == True:
                print('patch size={:d}, patch count={:d}x{:d}'.format(patch_size, patch_count_freq, patch_count_time))
                print('number of patches={:d}'.format(num_patches))

            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(patch_size, patch_size),
                                       stride=(patch_size, patch_size))
            if imagenet_pretrain == True:

                weight_sum = torch.sum(self.v.patch_embed.proj.weight, dim=1)

                new_proj.weight = torch.nn.Parameter(weight_sum.unsqueeze(1).repeat(1, 1, patch_size, patch_size))
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            if imagenet_pretrain == True:
                new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches,
                                                                            self.original_embedding_dim).transpose(1,
                                                                                                                   2).reshape(
                    1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)

                new_pos_embed = torch.nn.functional.interpolate(
                    new_pos_embed, size=(patch_count_freq, patch_count_time), mode='bilinear'
                )

                new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1, 2)

                self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
            else:

                new_pos_embed = nn.Parameter(
                    torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
                self.v.pos_embed = new_pos_embed
                trunc_normal_(self.v.pos_embed, std=.02)
    def adjust_positional_embedding(self, num_patches, imagenet_pretrain, patch_count):
        if imagenet_pretrain:
            original_num_patches = self.v.patch_embed.num_patches
            oringal_hw = int(original_num_patches ** 0.5)
            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(
                1, original_num_patches, self.original_embedding_dim
            ).transpose(1, 2).reshape(
                1, self.original_embedding_dim, oringal_hw, oringal_hw
            )
            new_h, new_w = patch_count
            new_pos_embed = torch.nn.functional.interpolate(
                new_pos_embed, size=(new_h, new_w), mode='bilinear'
            )
            new_pos_embed = new_pos_embed.reshape(
                1, self.original_embedding_dim, num_patches
            ).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat([
                self.v.pos_embed[:, :2, :].detach(), new_pos_embed
            ], dim=1))
        else:
            new_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + 2, self.original_embedding_dim)
            )
            self.v.pos_embed = new_pos_embed
            trunc_normal_(self.v.pos_embed, std=.02)
    @autocast()
    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        B = x.shape[0]
        if self.use_depatch:
            x, grid_size = self.v.patch_embed(x)
        else:
            x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2
        x = self.mlp_head(x)
        return x
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    input_freq = 128
    input_time = 256
    ast_mdl = ASTModel_Deformable(
        input_freq=input_freq,
        input_time=input_time,
        use_depatch=True,
        imagenet_pretrain=False
    ).to(device)
    test_input = torch.rand([10, input_freq, input_time]).to(device)
    test_output = ast_mdl(test_input)
    print("Output shape with DePatch:", test_output.shape)
    ast_mdl_standard = ASTModel_Deformable(
        input_freq=input_freq,
        input_time=input_time,
        use_depatch=False,
        imagenet_pretrain=False
    ).to(device)
    test_input = torch.rand([10, input_freq, input_time]).to(device)
    test_output_standard = ast_mdl_standard(test_input)
    print("Output shape without DePatch:", test_output_standard.shape)