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
        
        ## usar latent pa parameter predictor en un futuro
        
        # Decoder
        d3_out = self.trans_conv_3(latent)
        d2_out = self.trans_conv_2(d3_out)
        
        # add_layer_decoder (Inyección de skip connection) 
        sum_dec = d2_out + c1_out 
        
        mixed = self.mix_layer(sum_dec)
        reconstruction = self.trans_conv_1(mixed)
        
        return reconstruction, latent


## definimos aca el latentNCA, este es preliminar para probar
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
            nn.Conv2d(hidden_dims, channels, kernel_size=1) #, bias=False?
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
    
## definimos aca el dynamic latent nca con parametros que se dan, no entrenable  

## funcion para obtener filtros sobel para la percepción del NCA, esto es fijo y no entrenabl  
def get_sobel_kernel(channels):
    # Filtro Sobel X
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    # Filtro Sobel Y
    sobel_y = sobel_x.t()
    # Identidad
    identity = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
    
    # Combinamos y expandimos para procesar todos los canales (Depthwise)
    kernels = torch.stack([sobel_x, sobel_y, identity]) # Shape: (3, 3, 3)
    # Replicamos para cada canal de entrada
    kernels = kernels.unsqueeze(1).repeat(channels, 1, 1, 1) # Shape: (C*3, 1, 3, 3)
    return kernels

import torch.nn.functional as F

class DynamicLatentNCA(nn.Module):
    def __init__(self, channels=16, hidden_dims=64):
        super().__init__()
        self.channels = channels
        self.hidden_dims = hidden_dims
        
        # Percepción: Filtros Sobel fijos (no entrenables), register_buffer crea un tensor que no es un parámetro entrenable pero se mueve con el modelo (ej: a GPU)
        self.register_buffer('sobel_kernel', get_sobel_kernel(channels))
        
        # Normalización: Crucial para evitar que ReLU explote en la recurrencia
        self.norm = nn.InstanceNorm2d(channels * 3)
        
        # Único parámetro entrenable interno del NCA según el paper
        self.leak_factor = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, weights, steps=32):
        '''
    x: Tensor latente (B, C, H, W)
    weights: (w1_batch, b1_batch, w2_batch, b2_batch)
        '''    
        
        w1, b1, w2, b2 = weights
        B, C, H, W = x.shape
        H_dims = self.hidden_dims

        # Preparamos los pesos para la convolución agrupada
        # w1: [B, H_dims, C*3, 1, 1] -> [B*H_dims, C*3, 1, 1]
        w1 = w1.view(B * H_dims, C * 3, 1, 1)
        b1 = b1.view(B * H_dims)
        
        # w2: [B, C, H_dims, 1, 1] -> [B*C, H_dims, 1, 1]
        w2 = w2.view(B * C, H_dims, 1, 1)
        b2 = b2.view(B * C)

        for _ in range(steps):
            # 1. PERCEPCIÓN (Sobel)
            # Sigue siendo eficiente ya que usa groups=C 
            perceived = F.conv2d(x, self.sobel_kernel, padding=1, groups=C)
            perceived = self.norm(perceived) 

            # 2. UPDATE RULE (Dynamic Grouped Convolutions)
            # Paso A: Aplanamos el batch dentro de los canales para procesar todo junto
            # input: [1, B*(C*3), H, W]
            input_parallel = perceived.view(1, B * (C * 3), H, W)
            
            # Capa 1: Aplicamos B grupos. Cada grupo usa su propio set de pesos w1 
            dx = F.conv2d(input_parallel, weight=w1, bias=b1, groups=B)
            dx = F.relu(dx) # [1, B*H_dims, H, W]
            
            # Capa 2: Aplicamos B grupos de nuevo para volver a los canales originales 
            dx = F.conv2d(dx, weight=w2, bias=b2, groups=B)
            
            # Recomponemos el shape original [B, C, H, W]
            dx = dx.view(B, C, H, W)

            # 3. MORFOGÉNESIS
            # Aplicamos la actualización estocástica y el Leak Factor 
            mask = (torch.rand(B, 1, H, W, device=x.device) > 0.5).float()
            x = x + (self.leak_factor * dx * mask) 
            
        return x
    
## hacemos el parameter predictor  
class ParameterPredictor(nn.Module):
    def __init__(self, latent_dim, h_nca, out_nca):
        super().__init__()
        self.h_nca = h_nca
        self.out_nca = out_nca
        
        # Estas formas representan UN solo set de pesos
        self.w1_size = h_nca * (out_nca * 3) * 1 * 1
        self.b1_size = h_nca
        self.w2_size = out_nca * h_nca * 1 * 1
        self.b2_size = out_nca
        
        total_params = self.w1_size + self.b1_size + self.w2_size + self.b2_size
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, total_params)
        )

    def forward(self, e):
        # e: [Batch, latent_dim]
        p = self.net(e) 
        batch_size = p.shape[0]
        idx = 0
        
        # Extraemos y reformateamos manteniendo la dimensión del batch al principio
        # Pesos 1: [B, hidden, in*3, 1, 1]
        w1 = p[:, idx:idx + self.w1_size].view(batch_size, self.h_nca, self.out_nca * 3, 1, 1)
        idx += self.w1_size
        
        # Bias 1: [B, hidden]
        b1 = p[:, idx:idx + self.b1_size].view(batch_size, self.h_nca)
        idx += self.b1_size
        
        # Pesos 2: [B, out, hidden, 1, 1]
        w2 = p[:, idx:idx + self.w2_size].view(batch_size, self.out_nca, self.h_nca, 1, 1)
        idx += self.w2_size
        
        # Bias 2: [B, out]
        b2 = p[:, idx:idx + self.b2_size].view(batch_size, self.out_nca)
        
        return (w1, b1, w2, b2)
    

    
## integramos el autoencoder con el dynamic NCA en un modelo 
class MetaNCASegmenter(nn.Module):
    def __init__(self, ae_params, nca_steps=32):
        super().__init__()

        # Instanciamos el autoencoder
        self.ae = AutoEncoderDown3(ae_params)
        self.nca_steps = nca_steps
        
        # El NCA opera sobre los canales del espacio latente 
        latent_channels = ae_params['Conv2DParams3']['out_c']
        self.nca = DynamicLatentNCA(channels=latent_channels)

        # instanciamos el parameter predictor
        self.param_predictor = ParameterPredictor(latent_dim=latent_channels, h_nca=self.nca.hidden_dims, out_nca=latent_channels)

    def forward(self, x):
        # PASO 1: encoder
        # Ejecutamos las capas de tu encoder manualmente para guardar el 'skip_out'
        c1_out = self.ae.conv_layer_1(x) # Este es el skip que necesita el decoder final
        c2_out = self.ae.conv_layer_2(c1_out)
        
        p1_out = self.ae.pass_through_1(x)
        p2_out = self.ae.pass_through_2(p1_out)
        
        sum_enc = c2_out + p2_out 
        latent = self.ae.conv_layer_3(sum_enc)

        ## obtenemos los pesos dinámicos para el NCA a partir del espacio latente
        dynamic_weights = self.param_predictor(latent.mean(dim=[2,3]))  # Global Average Pooling para obtener un vector por muestra
        
        # PASO 2: EVOLUCIÓN NCA
        # El NCA refina el espacio latente
        latent_evolved = self.nca(latent, weights=dynamic_weights ,steps=self.nca_steps)
        
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

## crearemos un pool de entrenamiento para la estabilidad

class NCAPool:
    def __init__(self, pool_size, channels, h, w, device):
        # El pool guarda el estado latente completo (N, C, H, W) 
        self.size = pool_size
        self.pool = torch.zeros(pool_size, channels, h, w).to(device)
        self.device = device

    def sample(self, batch_size):
        # Seleccionamos índices al azar para el entrenamiento
        idx = torch.randint(0, self.size, (batch_size,))
        return self.pool[idx], idx

    def update(self, idx, new_states):
        # Guardamos los estados evolucionados de vuelta en el buffer 
        self.pool[idx] = new_states.detach()

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
