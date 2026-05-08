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
import torch.optim as optim

## AdjConv2D del paper
class AdjConv2D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, strides, padding, activation, batch_normalization, dropout_rate):
        super().__init__()
        layers = []
        # padding debe ser entero (ej: 1 para 3x3 'same')
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

## AdjConv2D Transpose del paper
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
    
## arquitectura autoencoder entera
class AutoEncoderDown3(nn.Module):
    def __init__(self, params):
        super().__init__()
        p = params
        
        # ENCODER
        # Camino Convolucional (Conv 1 -> Conv 2)
        self.conv_layer_1 = AdjConv2D(p['in_c'], **p['Conv2DParams1'])
        self.conv_layer_2 = AdjConv2D(p['Conv2DParams1']['out_c'], **p['Conv2DParams2'])
        
        # Camino Pass-through (Pass 1 -> Pass 2)
        self.pass_through_1 = AdjConv2D(p['in_c'], **p['PassThroughParams1'])
        self.pass_through_2 = AdjConv2D(p['PassThroughParams1']['out_c'], **p['PassThroughParams2'])
        
        # Capa final del Encoder (tras la suma)
        # La entrada aquí debe ser la suma de conv_layer_2 y pass_through_2
        self.conv_layer_3 = AdjConv2D(p['Conv2DParams2']['out_c'], **p['Conv2DParams3'])

        # DECODER 
        # Procesamiento del latente
        self.trans_conv_3 = AdjConv2DTranspose(p['Conv2DParams3']['out_c'], **p['Conv2DTransposeParams3'])
        self.trans_conv_2 = AdjConv2DTranspose(p['Conv2DTransposeParams3']['out_c'], **p['Conv2DTransposeParams2'])
        
        # Capa de Mezcla (Mix): Recibe (trans_conv_2 + skip_out)
        # skip_out es la salida de conv_layer_1
        self.mix_layer = AdjConv2D(p['Conv2DTransposeParams2']['out_c'], **p['MixParams'])
        
        # Salida Final
        self.trans_conv_1 = AdjConv2DTranspose(p['MixParams']['out_c'], **p['Conv2DTransposeParams1'])

    def forward(self, x):
        # Encoder
        c1_out = self.conv_layer_1(x) # Este es nuestro 'skip_out'
        c2_out = self.conv_layer_2(c1_out)
        
        p1_out = self.pass_through_1(x)
        p2_out = self.pass_through_2(p1_out)
        
        # add_layer_encoder
        sum_enc = c2_out + p2_out 
        latent = self.conv_layer_3(sum_enc)
        
        # Decoder
        d3_out = self.trans_conv_3(latent)
        d2_out = self.trans_conv_2(d3_out)
        
        # add_layer_decoder (Inyección de skip connection) 
        sum_dec = d2_out + c1_out 
        
        mixed = self.mix_layer(sum_dec)
        reconstruction = self.trans_conv_1(mixed)
        
        return reconstruction, latent


## definimos aca el latentNCA
class LatentNCA(nn.Module):
    def __init__(self, channels=16, hidden_dims=64):
        super().__init__()
        self.channels = channels
        
        # Percepción: Usamos una convolución agrupada para actuar como filtros locales
        # Esto es equivalente a que cada canal "vea" su vecindad
        self.perception = nn.Conv2d(channels, channels * 3, kernel_size=3, 
                                    padding=1, groups=channels, bias=False)
        
        # Regla de actualización: Un MLP (convoluciones 1x1)
        self.update_rule = nn.Sequential(
            nn.Conv2d(channels * 3, hidden_dims, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dims, channels, kernel_size=1)
        )
        
        # Inicialización: Empezamos con actualizaciones casi nulas para estabilidad
        nn.init.zeros_(self.update_rule[-1].weight)
        nn.init.zeros_(self.update_rule[-1].bias)

    def forward(self, x, steps=10):
        for _ in range(steps):
            # Percibir vecinos
            perceived = self.perception(x)
            # Calcular cambio
            delta = self.update_rule(perceived)
            # Aplicar actualización estocástica
            # Solo algunas células se actualizan en cada paso para fomentar robustez
            mask = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device) > 0.5).float()
            x = x + delta * mask
        return x
    

## integramos el autoencoder con el NCA en un modelo 
class NCASegmenter(nn.Module):
    def __init__(self, ae_params, nca_steps=32):
        super().__init__()

        # Instanciamos el autoencoder
        self.ae = AutoEncoderDown3(ae_params)
        self.nca_steps = nca_steps
        
        # El NCA opera sobre los canales del espacio latente 
        latent_channels = ae_params['Conv2DParams3']['out_c']
        self.nca = LatentNCA(channels=latent_channels)

    def forward(self, x):
        # PASO 1: encoder
        # Ejecutamos las capas de tu encoder manualmente para guardar el 'skip_out'
        c1_out = self.ae.conv_layer_1(x) # Este es el skip que necesita el decoder final
        c2_out = self.ae.conv_layer_2(c1_out)
        
        p1_out = self.ae.pass_through_1(x)
        p2_out = self.ae.pass_through_2(p1_out)
        
        sum_enc = c2_out + p2_out 
        latent = self.ae.conv_layer_3(sum_enc)
        
        # PASO 2: EVOLUCIÓN NCA
        # El NCA refina el espacio latente
        latent_evolved = self.nca(latent, steps=self.nca_steps)
        
        # PASO 3: DECODER
        # ejectuamos las capas del decoder manualmente para inyectar el skip connection
        d3_out = self.ae.trans_conv_3(latent_evolved)
        d2_out = self.ae.trans_conv_2(d3_out)
        
        # Reinyectamos el skip connection guardado en el paso 1
        sum_dec = d2_out + c1_out 
        
        mixed = self.ae.mix_layer(sum_dec)
        reconstruction = self.ae.trans_conv_1(mixed)
        
        return reconstruction, latent_evolved



## funcion referencia para entrenar autoencoder 
def train_ae(model, train_loader, epochs=10, device='cuda'):
    # congelamos el NCA
    for param in model.nca.parameters():
        param.requires_grad = False
        
    optimizer = optim.Adam(model.ae.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss() 
    
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for imgs, masks in train_loader:  # No necesitamos las máscaras para el AE
            imgs = imgs.to(device)
            masks = masks.to(device) # Debe ser (N, H, W) con valores {0, 1, 2}
            
            optimizer.zero_grad()
            
            # Forward: Obtenemos la reconstrucción
            outputs, _ = model.ae(imgs)
            
            loss = criterion(outputs, masks)  # Queremos que la salida se parezca a la entrada
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Época [{epoch+1}/{epochs}] - Loss: {total_loss/len(train_loader):.4f}")



## funcion de referencia para entrenar el modelo entrenado de forma entera
def train_nca(model, train_loader, epochs=10, device='cuda'):
    # CONGELAMOS EL AUTOENCODER
    # Solo queremos que aprenda la "regla de actualización" del NCA
    for param in model.ae.parameters():
        param.requires_grad = False
        
    # El optimizador solo ve los parámetros del NCA
    optimizer = optim.Adam(model.nca.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            
            optimizer.zero_grad()
            
            # Forward: Obtenemos la máscara predicha
            outputs, _ = model(imgs)
            
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Época [{epoch+1}/{epochs}] - Loss: {total_loss/len(train_loader):.4f}")
