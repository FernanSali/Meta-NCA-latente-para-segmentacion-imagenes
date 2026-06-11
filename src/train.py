import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn as nn 
from torch.utils.data import DataLoader
import torch.optim as optim

from src.model import AutoEncoderDown3, MetaNCASegmenter, NCASegmenter

from src.datadogs import dog_train_loader, dog_test_loader

import torch.nn.functional as F


## implementaremos una Dice loss multiclase
class MulticlassDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, weight=None):
        """
        smooth: Un valor muy pequeño para evitar la división por cero si 
                la intersección y la unión son nulas.
        weight: Pesos opcionales por clase (tensor de tamaño [num_classes])
        """
        super().__init__()
        self.smooth = smooth
        self.weight = weight

    def forward(self, logits, targets):
        """
        logits: Tensor de la red de forma (Batch, Num_Classes, H, W)
        targets: Tensor con las máscaras reales de forma (Batch, H, W) con índices enteros [0, Num_Classes-1]
        """
        # Convertir logits a probabilidades usando Softmax en la dimensión de los canales
        probs = F.softmax(logits, dim=1)
        num_classes = probs.shape[1]
        
        # Convertir targets (índices) a codificación One-Hot -> (Batch, Num_Classes, H, W)
        # Hacemos el permute para mover la dimensión de los canales al orden correcto
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # Colapsar las dimensiones espaciales (H, W) para operar vectorialmente por lote y clase
        # Quedan de forma: (Batch, Num_Classes, H*W)
        probs = probs.contiguous().view(probs.shape[0], num_classes, -1)
        targets_one_hot = targets_one_hot.contiguous().view(targets_one_hot.shape[0], num_classes, -1)
        
        # Calcular Intersección y Unión por cada muestra y cada clase
        intersection = torch.sum(probs * targets_one_hot, dim=-1)
        cardinality = torch.sum(probs + targets_one_hot, dim=-1)
        
        # Aplicar la fórmula del coeficiente Dice
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        
        # La pérdida es la inversa del coeficiente (queremos maximizar Dice, por ende minimizar 1 - Dice)
        dice_loss = 1.0 - dice_score
        
        # Promediar las pérdidas de las muestras del Batch
        # Queda un vector de tamaño [num_classes]
        class_losses = dice_loss.mean(dim=0)
        
        # Si hay pesos por clase asignados (ej: para darle más importancia al borde), los aplicamos
        if self.weight is not None:
            # Asegurar que el tensor de pesos esté en el mismo dispositivo de cómputo
            weight = self.weight.to(logits.device)
            class_losses = class_losses * (weight / weight.sum())
            return class_losses.sum()
        
        # Si no hay pesos, devolvemos el promedio simple de todas las clases
        return class_losses.mean()
    
## Y tambien una spatial continuity loss, (TV loss)
class SpatialContinuityLoss(nn.Module):
    def __init__(self, alpha=1e-4):
        """
        alpha: Factor de escala para controlar qué tan fuerte es la penalización.
               Un valor muy alto puede borrar bordes reales; un valor muy bajo no limpiará el ruido.
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, logits):
        """
        logits: Tensor flotante que escupe tu decoder, con forma (Batch, Classes, H, W)
        """
        # 1. Asegurar que operamos sobre probabilidades flotantes [0, 1]
        probs = F.softmax(logits, dim=1)
        
        # 2. Calcular la diferencia absoluta entre píxeles vecinos horizontales
        # Compara el píxel (x) con su vecino de la derecha (x+1)
        diff_h = torch.abs(probs[:, :, :, :-1] - probs[:, :, :, 1:])
        
        # 3. Calcular la diferencia absoluta entre píxeles vecinos verticales
        # Compara el píxel (y) con su vecino de abajo (y+1)
        diff_v = torch.abs(probs[:, :, :-1, :] - probs[:, :, 1:, :])
        
        # 4. Reducir a escalares sumando todas las diferencias de forma independiente
        # Al aplicar .sum() o .mean(), colapsamos las dimensiones espaciales y evitamos errores de tamaño
        loss_h = diff_h.mean()
        loss_v = diff_v.mean()
        
        # 5. Combinar las fuerzas y aplicar el factor alpha
        total_tv = loss_h + loss_v
        
        return self.alpha * total_tv



## archivo con ejemplos de entrenamientos para las 2 fases del modelo

## funcion referencia para entrenar autoencoder 
## antes hay que:
## ae_model = AutoEncoderDown3(ae_params).to(device)
def train_ae(ae_model, train_loader, epochs=10, device='cuda'):
    '''Entrenemiento del autoencoder solo, fase 1'''
    ae_model.to(device)

    criterion1 = nn.CrossEntropyLoss() # Ideal para las 3 clases del trimapa
    criterion2 = MulticlassDiceLoss() # Para mejorar la segmentación de bordes y detalles finos
    criterion3 = SpatialContinuityLoss(alpha=1e-1) # Para fomentar la continuidad espacial en las predicciones
    
    optimizer = optim.Adam(ae_model.parameters(), lr=1e-4)

    print(f"Iniciando Fase 1 directamente en el Autoencoder por {epochs} épocas...")

    for epoch in range(epochs):  # Número de épocas
        ae_model.train()  # Modo entrenamiento

        running_loss = 0.0


        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device).long() # Debe ser (N, H, W) con valores {0, 1, 2}

            # Forward
            # 'reconstruction' será tu predicción de máscara (N, 3, H, W)
            outputs, _ = ae_model(images) 

            # Suma de las 3 losses: Píxel (CE) + Región (Dice) + Suavizado (TV)
            loss_ce = criterion1(outputs, masks)
            loss_dice = criterion2(outputs, masks)
            loss_tv = criterion3(outputs)

            loss = loss_ce + loss_dice + loss_tv
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # calculamos promedio epoca
        print(f"Época [{epoch+1}/{epochs} - Loss promedio: {running_loss / len(train_loader):.4f})]")

        torch.cuda.empty_cache()

    # torch.save(ae_model.state_dict(), "")


## Pool de entrenamiento para estabilidad de los NCA
class NCAPool:
    '''pool de estados latentes para la estabilidad de los NCA'''
    def __init__(self, pool_size, channels, h, w, device, CHANNELS_LEVEL_1 = 64):
        # El pool guarda el estado latente completo (N, C, H, W) 
        self.size = pool_size
        self.pool = torch.zeros(pool_size, channels, h, w).to(device)
        self.device = device

        ## guardamos la salida original del encoder
        self.pool_base = torch.zeros(pool_size, channels, h, w, device=device)

        ## guarda las mascaras correspondientes al estado latente
        self.pool_masks = torch.zeros(pool_size, 4*h, 4*w, dtype=torch.long, device=device)

        ## guarda el c1
        self.pool_c1 = torch.zeros(pool_size, CHANNELS_LEVEL_1 , 4*h, 4*w, device=device)

    def sample(self, batch_size, current_latent, current_c1, current_masks):
        '''muestrea elemetos del pool de forma estocastica (95% viejos, 5% nuevos/reset)'''
        ## Seleccionamos índices al azar del pool
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)

        ## copiamos lo estados guardados en esos indices
        sampled_states = self.pool[idx].clone()
        sampled_bases = self.pool_base[idx].clone()
        sampled_masks = self.pool_masks[idx].clone()
        sampled_c1 = self.pool_c1[idx].clone()

        # Detectar qué elementos muestreados están completamente vacíos (en ceros)
        # Calculamos la suma absoluta a lo largo de las dimensiones de Canales, Alto y Ancho 
        # Si la suma es 0, el tensor está vacío.
        es_cero = (sampled_states.abs().sum(dim=[1, 2, 3]) == 0.0).float()
        # Reshapeamos a [B, 1, 1, 1] para poder multiplicar por los tensores latentes
        es_cero_mask = es_cero.view(batch_size, 1, 1, 1)
        ## 5 % prob de reserear la salida del encoder
        reset_mask = (torch.rand(batch_size, 1, 1, 1, device=self.device) < 0.05).float()

        # COMBINACIÓN DE MÁSCARAS:
        # Queremos forzar el uso de `current_latent` SI la máscara estocástica se activa 
        # O SI la ranura del pool está completamente vacía (es_cero_mask == 1).
        # Usamos la operación lógica OR mediante torch.max, o sumando y recortando en 1.0
        mascara_final_reset = torch.clamp(reset_mask + es_cero_mask, 0.0, 1.0)

        ## si reset_mask es 1 usamos el estado actual del encoder, si es 0 usamos el estado muestreado del pool
        x_input = current_latent * mascara_final_reset + sampled_states * (1.0 - mascara_final_reset)
        x_base = current_latent * mascara_final_reset + sampled_bases * (1.0 - mascara_final_reset)
        x_c1 = current_c1 * mascara_final_reset + sampled_c1 * (1.0 - mascara_final_reset)


        # Reshapeamos la máscara de reset para que se acople a las dimensiones (B, H, W) de las masks
        reset_mask_spatial = mascara_final_reset.squeeze(1) # Pasa de (B, 1, 1, 1) a (B, 1, 1)
        x_masks = current_masks * reset_mask_spatial.long() + sampled_masks * (1.0 - reset_mask_spatial).long()

        return x_input, x_base, x_c1, x_masks, idx

    def update(self, idx, new_states, original_bases, original_c1, final_masks):
        # Guardamos los estados evolucionados de vuelta en el buffer, sin gradietes
        self.pool[idx] = new_states.detach()
        self.pool_base[idx] = original_bases.detach()
        self.pool_c1[idx] = original_c1.detach()
        self.pool_masks[idx] = final_masks.detach()


## antes hay que:
## metanca_model = MetaNCASegmenter(ae_params=, nca_steps=).to(device)
## entrenar o cargar el autoencoder previamente entrenado en la fase 1
## metanca_model.ae.load_state_dict(torch.load(""))

def train_metanca(metanca_model, train_loader, epochs=10, device='cuda'):
    '''Entrenamiento del metaNCA segmenter, fase 2, con autoencoder congelado'''

    ## entrenamiento metaNCA segmenter
    print(" Congelando parámetros del Autoencoder...")
    for param in metanca_model.ae.parameters():
        param.requires_grad = False
        param.grad = None  # Forzamos la eliminación de cualquier gradiente residual de la Fase 1

    # Aseguramos que el Predictor y el NCA sí calculen gradientes
    for param in metanca_model.param_predictor.parameters():
        param.requires_grad = True
    metanca_model.nca.leak_factor.requires_grad = True


    # Pasamos única y exclusivamente los parámetros que requieren gradiente
    # Esto evita que Adam aplique updates o momentum en los tensores congelados
    optimizer = optim.Adam([
        {'params': [p for p in metanca_model.nca.parameters() if p.requires_grad]},
        {'params': [p for p in metanca_model.param_predictor.parameters() if p.requires_grad]}
    ], lr=1e-4)

    criterion1 = nn.CrossEntropyLoss() # Ideal para las 3 clases del trimapa
    criterion2 = MulticlassDiceLoss() # Para mejorar la segmentación de bordes y detalles finos
    #criterion3 = SpatialContinuityLoss(alpha=1e-4) # Para fomentar la continuidad espacial en las predicciones

    metanca_model.to(device)

    metanca_model.train()  # Activa modo entrenamiento general para el predictor
    metanca_model.ae.eval() # Fuerza al Autoencoder a mantenerse estático

    # Blindaje extra: Reemplazamos temporalmente el método train del AE 
    # para que ninguna llamada accidental en el loop altere sus sub-capas
    def dezafectar_train(mode=True):
        for module in metanca_model.ae.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.Dropout2d)):
                module.eval()
    metanca_model.ae.train = dezafectar_train
    dezafectar_train()


    ## inicializamos el pool latnete
    pool_latente = NCAPool(pool_size=512, channels=16, h=64, w=64, device=device)
    print(" IIniciando loop de prueba preliminar blindado...")

    for epoch in range(epochs):
        total_loss = 0.0

        for i, (imgs, masks) in enumerate(train_loader):
            imgs, masks = imgs.to(device), masks.to(device)
            
            # set_to_none=True elimina los gradientes del optimizador liberando VRAM
            optimizer.zero_grad(set_to_none=True)
            
            # Obtenemos el latente base inicial y el embedding para los pesos dinámicos
            with torch.no_grad():
                c1_out = metanca_model.ae.conv_layer_1(imgs)
                c2_out = metanca_model.ae.conv_layer_2(c1_out)
                p1_out = metanca_model.ae.pass_through_1(imgs)
                p2_out = metanca_model.ae.pass_through_2(p1_out)

                sum_enc = c2_out + p2_out
                latent_base = metanca_model.ae.conv_layer_3(sum_enc)

            
            # MUESTREO DEL POOL
            # En lugar de usar siempre 'latent_base', dejamos que el pool decida estocásticamente
            latent_input, original_bases, original_c1, target_masks, pool_indices = pool_latente.sample(imgs.shape[0], latent_base, c1_out, masks)

            # Generamos los pesos dinámicos a partir del lote actual
            latent_vector_limpio = original_bases.mean(dim=[2, 3])  ## simulacion adn pr ahora
            dynamic_weights = metanca_model.param_predictor(latent_vector_limpio)
            
            
            # El NCA evoluciona el estado seleccionado (ya sea inicial o intermedio del pool)
            ## ocupamos pasos aleatorios

            if epoch <= 3:
                pasos_tensor = torch.randint(low=8, high=16 + 1, size=(1,))
                steps_nca = pasos_tensor.item()
            else:
                pasos_tensor = torch.randint(low=8, high=32 + 1, size=(1,))
                steps_nca = pasos_tensor.item()
    
            latent_evolved = metanca_model.nca(latent_input, weights=dynamic_weights, steps=steps_nca)
            
            # ACTUALIZACIÓN DEL POOL
            # Guardamos los estados resultantes en el pool para la siguiente oportunidad
            pool_latente.update(pool_indices, latent_evolved, original_bases, original_c1, target_masks)

            
            # DECODER FINAL Y PÉRDIDA
            # El decoder toma los estados evolucionados para proyectar a la máscara final
            d3_out = metanca_model.ae.trans_conv_3(latent_evolved)
            d2_out = metanca_model.ae.trans_conv_2(d3_out)

            sum_dec = d2_out + original_c1  # Inyección del skip connection original

            mixed = metanca_model.ae.mix_layer(sum_dec)
            outputs = metanca_model.ae.trans_conv_1(mixed)
            
            ## optimizacion clasica
            # Suma de las 3 losses: Píxel (CE) + Región (Dice) + Suavizado (TV)
            loss_ce = criterion1(outputs, target_masks)
            loss_dice = criterion2(outputs, target_masks)
            #loss_tv = criterion3(outputs)

            loss = loss_ce + loss_dice #+ loss_tv

            loss.backward()

            # Escudo anti-explosión final
            #torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, metanca_model.parameters()), max_norm=1.0)

            optimizer.step()
            
            # Restricción matemática de estabilidad del autómata celular
            with torch.no_grad():
                metanca_model.nca.leak_factor.clamp_(1e-3, 1e-1)
            
            total_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(f"\n✅ Van {i + 1} batches procesados exitosamente.")
                print(f"📊 Loss promedio actual: {total_loss / (i + 1):.4f}")
                print(f"💧 Leak Factor actual: {metanca_model.nca.leak_factor.item():.4f}")

        print(f"\n🎯 Época [{epoch+1}/{epochs}] - Loss promedio: {total_loss / len(train_loader):.4f}")
            
            
    # lo guardamos al final
    #torch.save(metanca_model.state_dict(), "meta_nca2.pth")




## Esta funcion era para el nca preliminar solo, sin parametros computados sino entrenables
def train_nca(nca_model, train_loader, epochs=10, device='cuda'):
    # CONGELAMOS EL AUTOENCODER
    # Solo queremos que aprenda la "regla de actualización" del NCA
    for param in nca_model.ae.parameters():
        param.requires_grad = False
        
    # El optimizador solo ve los parámetros del NCA
    optimizer = optim.Adam(nca_model.nca.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    nca_model.to(device)

    for epoch in range(epochs):
        nca_model.train()
        total_loss = 0
        
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            
            optimizer.zero_grad()
            
            # Forward: Obtenemos la máscara predicha
            outputs, _ = nca_model(imgs)
            
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Época [{epoch+1}/{epochs}] - Loss: {total_loss/len(train_loader):.4f}")