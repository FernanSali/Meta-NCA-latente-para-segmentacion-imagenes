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
        
        self.register_buffer('sobel_kernel', get_sobel_kernel(channels))
        
        # Ajustamos a channels * 3 para acoplar la percepción Sobel
        self.norm = nn.InstanceNorm2d(channels * 3, affine=False)
        self.leak_factor = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, weights, steps=32):
        '''
        x: Tensor latente (B, C, H, W) 
        weights: Tupla (w1, b1, w2, b2) generada por el ParameterPredictor 
        '''    
        w1, b1, w2, b2 = weights
        B, C, H, W = x.shape

        for _ in range(steps):
            # 1. PERCEPCIÓN (Sobel + InstanceNorm para cortar la explosión recurrente) 
            perceived = F.conv2d(x, self.sobel_kernel, padding=1, groups=C)
            perceived = self.norm(perceived) 

            # 2. UPDATE RULE (Convolución dinámica funcional 1x1 limpia por muestra) 
            dx_list = []
            for b in range(B):
                # Procesamos muestra por muestra para mantener el grafo de gradientes ligero
                # perceived[b:b+1] -> (1, C*3, H, W)
                # w1[b] -> (H_dims, C*3, 1, 1) | b1[b] -> (H_dims)
                out_layer1 = F.conv2d(perceived[b:b+1], weight=w1[b], bias=b1[b], stride=1, padding=0)
                out_layer1 = F.relu(out_layer1) 
                
                # w2[b] -> (C, H_dims, 1, 1) | b2[b] -> (C)
                out_layer2 = F.conv2d(out_layer1, weight=w2[b], bias=b2[b], stride=1, padding=0)
                dx_list.append(out_layer2)
            
            # Unimos el lote de nuevo sin romper la memoria interna
            dx = torch.cat(dx_list, dim=0)

            # 3. MORFOGÉNESIS (Actualización estocástica) 
            mask = (torch.rand(B, 1, H, W, device=x.device) > 0.5).float() 
            x = x + (self.leak_factor * dx * mask) 
            
        return x
    
## hacemos el parameter predictor  
class ParameterPredictor(nn.Module):
    def __init__(self, latent_dim, h_nca, out_nca):
        super().__init__()
        self.h_nca = h_nca
        self.out_nca = out_nca
        
        self.w1_size = h_nca * (out_nca * 3) * 1 * 1
        self.b1_size = h_nca
        self.w2_size = out_nca * h_nca * 1 * 1
        self.b2_size = out_nca
        
        total_params = self.w1_size + self.b1_size + self.w2_size + self.b2_size
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, total_params)
        )

    def forward(self, e):
        # e: [Batch, latent_dim] 
        p = self.net(e) 
        batch_size = p.shape[0]
        
        # Desempaquetamos de forma nativa manteniendo la separación del lote desde el inicio
        w1 = p[:, :self.w1_size].contiguous().view(batch_size, self.h_nca, self.out_nca * 3, 1, 1)
        idx = self.w1_size
        
        b1 = p[:, idx:idx + self.b1_size].contiguous().view(batch_size, self.h_nca)
        idx += self.b1_size
        
        w2 = p[:, idx:idx + self.w2_size].contiguous().view(batch_size, self.out_nca, self.h_nca, 1, 1)
        idx += self.w2_size
        
        b2 = p[:, idx:].contiguous().view(batch_size, self.out_nca)
        
        return (w1, b1, w2, b2)
    

    
## integramos el autoencoder con el dynamic NCA en un modelo 
class MetaNCASegmenter(nn.Module):
    def __init__(self, ae_params, nca_steps=32):
        super().__init__()
        self.ae = AutoEncoderDown3(ae_params)
        self.nca_steps = nca_steps
        
        latent_channels = ae_params['Conv2DParams3']['out_c']
        self.nca = DynamicLatentNCA(channels=latent_channels)
        self.param_predictor = ParameterPredictor(latent_dim=latent_channels, h_nca=self.nca.hidden_dims, out_nca=latent_channels)

    def forward(self, x, steps=None):
        current_steps = steps if steps is not None else self.nca_steps
        
        # PASO 1: Encoder 
        c1_out = self.ae.conv_layer_1(x) 
        c2_out = self.ae.conv_layer_2(c1_out)
        p1_out = self.ae.pass_through_1(x)
        p2_out = self.ae.pass_through_2(p1_out)
        
        sum_enc = c2_out + p2_out 
        latent = self.ae.conv_layer_3(sum_enc)

        # Global Average Pooling para alimentar al predictor 
        latent_vector = latent.mean(dim=[2, 3]) 
        dynamic_weights = self.param_predictor(latent_vector) 
        
        # PASO 2: Evolución NCA en el espacio latente 
        latent_evolved = self.nca(latent, weights=dynamic_weights, steps=current_steps)
        
        # PASO 3: Decoder 
        d3_out = self.ae.trans_conv_3(latent_evolved)
        d2_out = self.ae.trans_conv_2(d3_out)
        
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

## funcion de referencia para entrenar el modelo entrenado de forma entera, este el nca normal
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

## funcion entrenamiento preliminar para entrenar meta nca  (la fase 2, entrenar el nca dinamico, es decir el predictor de parametros)

def train_meta_nca(model, train_loader, device='cuda'):
    '''entrenamiento metaNCA, 1 epoch'''
    ## entrenamiento metaNCA segmenter
    print(" Congelando parámetros del Autoencoder...")
    for param in model.ae.parameters():
        param.requires_grad = False
        param.grad = None  # Forzamos la eliminación de cualquier gradiente residual de la Fase 1

    # Aseguramos que el Predictor y el NCA sí calculen gradientes
    for param in model.param_predictor.parameters():
        param.requires_grad = True
    model.nca.leak_factor.requires_grad = True


    # Pasamos única y exclusivamente los parámetros que requieren gradiente
    # Esto evita que Adam aplique updates o momentum en los tensores congelados
    optimizer = optim.Adam([
        {'params': [p for p in model.nca.parameters() if p.requires_grad]},
        {'params': [p for p in model.param_predictor.parameters() if p.requires_grad]}
    ], lr=1e-3)

    criterion = nn.CrossEntropyLoss()
    model.to(device)

    model.train()  # Activa modo entrenamiento general para el predictor
    model.ae.eval() # Fuerza al Autoencoder a mantenerse estático

    # Blindaje extra: Reemplazamos temporalmente el método train del AE 
    # para que ninguna llamada accidental en el loop altere sus sub-capas
    def dezafectar_train(mode=True):
        for module in model.ae.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.Dropout2d)):
                module.eval()
    model.ae.train = dezafectar_train
    dezafectar_train()


    total_loss = 0
    print(" Iniciando loop de prueba preliminar blindado...")

    for i, (imgs, masks) in enumerate(train_loader):
        imgs, masks = imgs.to(device), masks.to(device)
        
        # set_to_none=True elimina los gradientes del optimizador liberando VRAM
        optimizer.zero_grad(set_to_none=True)
        
        # Forward: Obtención de la salida a través del meta-NCA latente
        outputs, _ = model(imgs)
        
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        # Restricción matemática de estabilidad del autómata celular
        with torch.no_grad():
            model.nca.leak_factor.clamp_(1e-3, 1e3)
        
        total_loss += loss.item()

        if (i + 1) % 100 == 0:
            print(f"\n✅ Van {i + 1} batches procesados exitosamente.")
            print(f"📊 Loss promedio actual: {total_loss / (i + 1):.4f}")
            print(f"💧 Leak Factor actual: {model.nca.leak_factor.item():.4f}")
        