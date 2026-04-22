'''
# Crear el ambiente
python -m venv venv

# Activarlo (Windows)
.\venv\Scripts\activate


## para dar permisos para activarlo
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

## crear archivo para requirements
pip freeze > requirements.txt
'''
from torch.utils.data import Dataset
import torch
import torch.nn as nn 

class AdjConv2D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, strides, padding, activation, batch_normalization, dropout_rate):
        super().__init__()
        layers = []
        # En PyTorch el padding debe ser entero (ej: 1 para 3x3 'same')
        layers.append(nn.Conv2d(in_c, out_c, kernel_size, stride=strides, padding=padding))
        
        if batch_normalization:
            layers.append(nn.BatchNorm2d(out_c))
        
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
            
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(p=dropout_rate))
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)

class AdjConv2DTranspose(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, strides, padding, activation, batch_normalization, dropout_rate):
        super().__init__()
        layers = []
        # Se añade output_padding=1 si strides=2 para recuperar dimensiones pares
        out_pad = 1 if strides > 1 else 0 
        layers.append(nn.ConvTranspose2d(in_c, out_c, kernel_size, stride=strides, 
                                         padding=padding, output_padding=out_pad))
        
        if batch_normalization:
            layers.append(nn.BatchNorm2d(out_c))
            
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())

        if dropout_rate > 0:
            layers.append(nn.Dropout2d(p=dropout_rate))
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)
    

class AutoEncoderDown3(nn.Module):
    def __init__(self, params):
        super().__init__()
        p = params
        
        # --- ENCODER ---
        # Camino Convolucional (Conv 1 -> Conv 2)
        self.conv_layer_1 = AdjConv2D(p['in_c'], **p['Conv2DParams1'])
        self.conv_layer_2 = AdjConv2D(p['Conv2DParams1']['out_c'], **p['Conv2DParams2'])
        
        # Camino Pass-through (Pass 1 -> Pass 2)
        self.pass_through_1 = AdjConv2D(p['in_c'], **p['PassThroughParams1'])
        self.pass_through_2 = AdjConv2D(p['PassThroughParams1']['out_c'], **p['PassThroughParams2'])
        
        # Capa final del Encoder (tras la suma)
        # La entrada aquí debe ser la suma de conv_layer_2 y pass_through_2
        self.conv_layer_3 = AdjConv2D(p['Conv2DParams2']['out_c'], **p['Conv2DParams3'])

        # --- DECODER ---
        # Procesamiento del latente
        self.trans_conv_3 = AdjConv2DTranspose(p['Conv2DParams3']['out_c'], **p['Conv2DTransposeParams3'])
        self.trans_conv_2 = AdjConv2DTranspose(p['Conv2DTransposeParams3']['out_c'], **p['Conv2DTransposeParams2'])
        
        # Capa de Mezcla (Mix): Recibe (trans_conv_2 + skip_out)
        # skip_out es la salida de conv_layer_1
        self.mix_layer = AdjConv2D(p['Conv2DTransposeParams2']['out_c'], **p['MixParams'])
        
        # Salida Final
        self.trans_conv_1 = AdjConv2DTranspose(p['MixParams']['out_c'], **p['Conv2DTransposeParams1'])

    def forward(self, x):
        # 1. Encoder
        c1_out = self.conv_layer_1(x) # Este es nuestro 'skip_out'
        c2_out = self.conv_layer_2(c1_out)
        
        p1_out = self.pass_through_1(x)
        p2_out = self.pass_through_2(p1_out)
        
        # add_layer_encoder
        sum_enc = c2_out + p2_out 
        latent = self.conv_layer_3(sum_enc)
        
        # 2. Decoder
        d3_out = self.trans_conv_3(latent)
        d2_out = self.trans_conv_2(d3_out)
        
        # add_layer_decoder (Inyección de skip connection) 
        sum_dec = d2_out + c1_out 
        
        mixed = self.mix_layer(sum_dec)
        reconstruction = self.trans_conv_1(mixed)
        
        return reconstruction, latent













print("Corrido sin errrores")