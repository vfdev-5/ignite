import torch
import torch.nn as nn


from efficientnet_pytorch import EfficientNet, get_model_params
from efficientnet_pytorch.model import MBConvBlock
from efficientnet_pytorch.utils import Conv2dStaticSamePadding, MemoryEfficientSwish, round_filters, round_repeats


def create_stem_unit(in_channels, num_features):
    stem_conv = Conv2dStaticSamePadding(in_channels, num_features, image_size=224, kernel_size=3, stride=2, bias=False)
    stem_bn = nn.BatchNorm2d(num_features, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
    return nn.Sequential(stem_conv, stem_bn)


def create_block_unit(block_args, global_params):
    blocks = []
    # Update block input and output filters based on depth multiplier.
    block_args = block_args._replace(
        input_filters=round_filters(block_args.input_filters, global_params),
        output_filters=round_filters(block_args.output_filters, global_params),
        num_repeat=round_repeats(block_args.num_repeat, global_params)
    )    

    # The first block needs to take care of stride and filter size increase.
    blocks.append(MBConvBlock(block_args, global_params))
    if block_args.num_repeat > 1:
        block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
    for _ in range(block_args.num_repeat - 1):
        blocks.append(MBConvBlock(block_args, global_params))  
    
    return blocks


def create_head(in_channels, out_channels, global_params):
    # Batch norm parameters
    bn_mom = 1 - global_params.batch_norm_momentum
    bn_eps = global_params.batch_norm_epsilon        
    conv_head = Conv2dStaticSamePadding(in_channels, out_channels, image_size=224, kernel_size=1, bias=False)
    bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
    return nn.Sequential(conv_head, bn1)


def replicate(old_module, new_module):
#     new_module.load_state_dict(old_module.state_dict())
    pass


class GrowingEffNetB0(nn.Module):
    
    def __init__(self, in_channels, num_classes, reduce=4):
        super(GrowingEffNetB0, self).__init__()
                
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_features = 32 // reduce
        self.stem = nn.ModuleList([
            create_stem_unit(in_channels, self.num_features),
        ])
        self.swish = MemoryEfficientSwish()

        # Build blocks
        self.blocks_args, self.global_params = get_model_params('efficientnet-b0', {"depth_divisor": 4})
        # Create smaller block
        for i, bargs in enumerate(self.blocks_args):
            self.blocks_args[i] = bargs._replace(
                input_filters=bargs.input_filters // reduce,
                output_filters=bargs.output_filters // reduce                
            )

        blocks = []
        for block_args in self.blocks_args:
            blocks += create_block_unit(block_args, self.global_params)
                
        self.blocks = nn.ModuleList([nn.ModuleList([b, ]) for b in blocks])

        # Head
        self.head_in_channels = block_args.output_filters  # output of final block
        self.head_out_channels = round_filters(1280 // reduce, self.global_params)
        self.head = nn.ModuleList([
            create_head(self.head_in_channels, self.head_out_channels, self.global_params),
        ])

        # Final linear layer
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(self.global_params.dropout_rate)
        self.fc = nn.ModuleList([
            nn.Linear(self.head_out_channels, self.num_classes),
        ])
        
    def _grow_stem(self, device):
        new_stem_unit = create_stem_unit(self.in_channels, self.num_features)
        new_stem_unit = new_stem_unit.to(device)
        replicate(self.stem[-1], new_stem_unit)
        self.stem.append(new_stem_unit)
    
    def _grow_blocks(self, device):
        new_blocks = []
        for block_args in self.blocks_args:
            new_blocks += create_block_unit(block_args, self.global_params)
                
        for block, new_block_unit in zip(self.blocks, new_blocks):            
            new_block_unit = new_block_unit.to(device)
            replicate(new_block_unit, block[-1])
            block.append(new_block_unit)
    
    def _grow_head(self, device):
        new_head_unit = create_head(self.head_in_channels, self.head_out_channels, self.global_params)
        new_head_unit = new_head_unit.to(device)
        replicate(self.head[-1], new_head_unit)
        self.head.append(new_head_unit)
        
    def _grow_linear(self, device):
        new_fc = nn.Linear(self.head_out_channels, self.num_classes)
        new_fc = new_fc.to(device)
        replicate(self.fc[-1], new_fc)
        self.fc.append(new_fc)

    def _forward_module_sum(self, x, module_list, **kwargs):
        x_out = []
        for unit in module_list:
            x_out.append(unit(x, **kwargs))
        
        x = torch.stack(x_out)
        x = torch.sum(x, dim=0)
        return x
    
    def _forward_stem(self, x):
        x = self._forward_module_sum(x, self.stem)
        x = self.swish(x)
        return x
    
    def _forward_block(self, x, block):        
        drop_connect_rate = self.global_params.drop_connect_rate
        x = self._forward_module_sum(x, block, drop_connect_rate=drop_connect_rate)
        return x    

    def _forward_blocks(self, x):
        for block in self.blocks:
            x = self._forward_block(x, block)
        return x
    
    def _forward_head(self, x):
        x = self._forward_module_sum(x, self.head)
        x = self.swish(x)
        return x
    
    def _forward_linear(self, x):
        bs = x.shape[0]
        # Pooling and final linear layer
        x = self.avg_pooling(x)
        x = x.view(bs, -1)
        x = self.dropout(x)
        x = self._forward_module_sum(x, self.fc)
        return x        

    def forward(self, x):        
        x = self._forward_stem(x)        
        x = self._forward_blocks(x)
        x = self._forward_head(x)
        x = self._forward_linear(x)
        return x

    def grow(self, device):
        self._grow_stem(device)
        self._grow_blocks(device)
        self._grow_head(device)
        self._grow_linear(device)
