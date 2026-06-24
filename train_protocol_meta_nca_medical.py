import torch

from src import model as models


import wandb


# Configuración de hardware acelerado
device = (
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available() 
    else "cpu"
)

print(f"Usando el dispositivo: {device}")


## data y data loader 

# imports:
import os
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import nibabel as nib
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

class SliceDataset(Dataset):
    """Pytorch Dataset for loading preprocessed 2D slices"""
    def __init__(self, image_dir, label_dir):
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".npy")])
        self.image_dir = image_dir
        self.label_dir = label_dir

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fname = self.image_files[idx]
        img = np.load(os.path.join(self.image_dir, fname))
        lbl = np.load(os.path.join(self.label_dir, fname))
        
        assert img.shape == lbl.shape, f"Shape mismatch in {fname}: {img.shape} vs {lbl.shape}"
        assert img.shape == (320, 320), f"Unexpected slice size in {fname}: {img.shape}"

        img = torch.from_numpy(img).unsqueeze(0).float()
        lbl = torch.from_numpy(lbl).unsqueeze(0).float()

        return img, lbl
    
## instanciamos 
dataset = SliceDataset("Data/prostate_train/images", "Data/prostate_train/labels")


## Borramos instancias con labels vacios
print(f"Cantidad inicial de slices: {len(dataset.image_files)}")

# 1. Crear una nueva lista con los archivos que sí contienen etiquetas
archivos_limpios = []

for fname in dataset.image_files:
    # Construimos la ruta usando la carpeta guardada en el dataset
    lbl_path = os.path.join(dataset.label_dir, fname)
    lbl = np.load(lbl_path)
    
    # Si tiene contenido, guardamos el nombre del archivo
    if np.mean(lbl) > 0:
        archivos_limpios.append(fname)

# 2. Reemplazar la lista interna del dataset por la lista filtrada
dataset.image_files = archivos_limpios

print(f"Cantidad final de slices (limpios): {len(dataset.image_files)}")

## dataloader
batch_size = 4 # Ajusta según tu memoria de video (VRAM)

prostate_train_loader = DataLoader(
    dataset, 
    batch_size=batch_size, 
    shuffle=True,      
    #num_workers=2,     # Acelera la carga de datos
    #pin_memory=True    # Mejora velocidad de transferencia a GPU
)

## definimos las funciones de perdida y metricas

def dice_loss(pred, target, smooth=1e-5):
    """Dice loss for segmentation"""
    pred = torch.sigmoid(pred)
    pred = torch.flatten(pred)
    target = torch.flatten(target)
    intersect = (pred * target).sum()
    dice = (2 * intersect + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice

def dice_score(pred, target, smooth=1e-5):
    """Dice coefficient metric"""
    pred = torch.sigmoid(pred)
    pred = torch.flatten(pred > 0.5).float()
    target = torch.flatten(target)
    intersect = (pred * target).sum()
    score = (2 * intersect + smooth) / (pred.sum() + target.sum() + smooth)
    return score

def bce_loss(pred, target):
    """Binary cross-entropy loss"""
    return F.binary_cross_entropy_with_logits(pred, target)

## defininimos los parametros

# 1. Configuración de Canales (Flujo Simétrico de Dimensiones)
CHANNELS_IN       = 1   # Imágenes RGB de entrada
CHANNELS_LEVEL_1  = 32  # Resolución alta (Skip Connection c1_out / d2_out)
CHANNELS_LEVEL_2  = 64  # Resolución intermedia (Salida de compresión y Pass-Through)
CHANNELS_LATENT   = 32  # Cuello de botella latente (Espacio del ADN / Entrada del NCA)
CHANNELS_OUT      = 1   # Clases del label

# 2. Parámetros de Convolución Estándar
KERNEL_STD        = 3
STRIDE_NORMAL     = 1
STRIDE_COMPRESS   = 2
PADDING_STD       = 1

# 3. Regularización y Normalización
DROPOUT_ENCODER   = 0.1
DROPOUT_DECODER   = 0.0
ACT_RELU          = 'relu'
ACT_NONE          = None


## DICCIONARIO DE PARÁMETROS CONFIGURADO CON VARIABLES COMUNES
params_encoder = {
    'in_c': CHANNELS_IN,
    
    # ENCODER 
    # Conv1 define el canal de nuestro skip_out de alta resolución.
    'Conv2DParams1': {
        'out_c': CHANNELS_LEVEL_1, 
        'kernel_size': KERNEL_STD, 
        'strides': STRIDE_NORMAL, 
        'padding': PADDING_STD, 
        'activation': ACT_RELU, 
        'batch_normalization': True, 
        'dropout_rate': DROPOUT_ENCODER
    },
    
    # Conv2 comprime el espacio (/2) y sube canales.
    'Conv2DParams2': {
        'out_c': CHANNELS_LEVEL_2, 
        'kernel_size': KERNEL_STD, 
        'strides': STRIDE_COMPRESS, 
        'padding': PADDING_STD, 
        'activation': ACT_RELU, 
        'batch_normalization': True, 
        'dropout_rate': DROPOUT_ENCODER
    },
    
    # Camino Pass-through: Debe igualar exactamente la salida de Conv2DParams2 (Canales e impacto de Stride)
    'PassThroughParams1': {
        'out_c': CHANNELS_LEVEL_1, 
        'kernel_size': 1, 
        'strides': STRIDE_NORMAL, 
        'padding': 0, 
        'activation': ACT_RELU, 
        'batch_normalization': True, 
        'dropout_rate': 0.0
    },
    'PassThroughParams2': {
        'out_c': CHANNELS_LEVEL_2, 
        'kernel_size': KERNEL_STD, 
        'strides': STRIDE_COMPRESS, 
        'padding': PADDING_STD, 
        'activation': ACT_RELU, 
        'batch_normalization': True, 
        'dropout_rate': 0.0
    },
    
    # Capa final del Encoder (Conv3): Colapsamos la información al espacio latente objetivo.
    'Conv2DParams3': {
        'out_c': CHANNELS_LATENT, 
        'kernel_size': KERNEL_STD, 
        'strides': STRIDE_COMPRESS, 
        'padding': PADDING_STD, 
        'activation': ACT_RELU, 
        'batch_normalization': True, 
        'dropout_rate': DROPOUT_ENCODER
    },

    # ─── DECODER ──────────────────────────────────────────────────────────
    # TransConv3 procesa el latente del NCA y recupera la resolución espacial (*2).
    'Conv2DTransposeParams3': {
        'out_c': CHANNELS_LEVEL_2, 
        'kernel_size': KERNEL_STD, 
        'strides': STRIDE_COMPRESS, 
        'padding': PADDING_STD, 
        'activation': ACT_RELU, 
        'batch_normalization': True, 
        'dropout_rate': DROPOUT_DECODER
    },
    
    # TransConv2 recupera la resolución original (*2). 
    # CRÍTICO: Debe salir con los mismos canales del skip_out (CHANNELS_LEVEL_1).
    'Conv2DTransposeParams2': {
        'out_c': CHANNELS_LEVEL_1, 
        'kernel_size': KERNEL_STD, 
        'strides': STRIDE_COMPRESS, 
        'padding': PADDING_STD, 
        'activation': ACT_RELU, 
        'batch_normalization': True, 
        'dropout_rate': DROPOUT_DECODER
    },
    
    # MixParams recibe la suma directa de d2_out + c1_out. Procesa la combinación manteniendo el ancho.
    'MixParams': {
        'out_c': CHANNELS_LEVEL_1, 
        'kernel_size': KERNEL_STD, 
        'strides': STRIDE_NORMAL, 
        'padding': PADDING_STD, 
        'activation': ACT_RELU, 
        'batch_normalization': True, 
        'dropout_rate': DROPOUT_DECODER
    },
    
    # Salida final del Decoder: Proyecta los canales refinados a los canales del trimapa.
    'Conv2DTransposeParams1': {
        'out_c': CHANNELS_OUT, 
        'kernel_size': KERNEL_STD, 
        'strides': STRIDE_NORMAL, 
        'padding': PADDING_STD, 
        'activation': ACT_NONE, 
        'batch_normalization': False, 
        'dropout_rate': DROPOUT_DECODER
    }
}

## creamos la clase del ncapool 

## Pool de entrenamiento para estabilidad de los NCA
class NCAPool:
    '''pool de estados latentes para la estabilidad de los NCA'''
    def __init__(self, pool_size, channels, h, w, device, CHANNELS_LEVEL_1 = CHANNELS_LEVEL_1):
        # El pool guarda el estado latente completo (N, C, H, W) 
        self.size = pool_size
        self.pool = torch.zeros(pool_size, channels, h, w).to(device)
        self.device = device

        ## guardamos la salida original del encoder
        self.pool_base = torch.zeros(pool_size, channels, h, w, device=device)

        ## guarda las mascaras correspondientes al estado latente
        self.pool_masks = torch.zeros(pool_size, 4*h, 4*w, dtype=torch.float32, device=device)

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
        reset_mask_spatial = mascara_final_reset #.squeeze(1) # Pasa de (B, 1, 1, 1) a (B, 1, 1)

        # Le agregamos la dimensión de canales a sampled_masks para que sea [4, 1, 320, 320]
        sampled_masks_4d = sampled_masks.unsqueeze(1)
        
        x_masks = current_masks * reset_mask_spatial.long() + sampled_masks_4d * (1.0 - reset_mask_spatial).long()
        

        return x_input, x_base, x_c1, x_masks.squeeze(1), idx

    def update(self, idx, new_states, original_bases, original_c1, final_masks):
        # Guardamos los estados evolucionados de vuelta en el buffer, sin gradietes
        self.pool[idx] = new_states.detach()
        self.pool_base[idx] = original_bases.detach()
        self.pool_c1[idx] = original_c1.detach()
        self.pool_masks[idx] = final_masks.detach()




def main():

    # Inicializamos W&B absorbiendo la metadata estructural en el config
    wandb.init(
        entity="fsalinab-pontificia-universidad-cat-lica-de-chile",
        project="medical_meta_nca",
        name="meta_nca_medical_v1",
        config={
            "learning_rate": 1e-4,
            "epochs": 50,
            "batch_size": len(prostate_train_loader.dataset) // len(prostate_train_loader),  # Ajuste automático según el tamaño del dataset    

            "CHANNELS_IN": CHANNELS_IN,         # canales de entrada RGB (3) o gray scale (1)
            "CHANNELS_LEVEL_1": CHANNELS_LEVEL_1, # dimension skip connection
            "CHANNELS_LEVEL_2": CHANNELS_LEVEL_2, # dimension intermedia del encoder
            "CHANNELS_LATENT": CHANNELS_LATENT,      # canales del Nca (ADN)
            "CHANNELS_OUT": CHANNELS_OUT,       # canales de salida (categorias del label)

            "nca_hidden_dims": 64,  # Dimensión interna del NCA para mlp
            "min_nca_steps": 8,
            "max_nca_steps": 32,
            "pool_size": 512,

            ## los otros parametros
            
            "KERNEL_STD"        : KERNEL_STD,
            "STRIDE_NORMAL"     : STRIDE_NORMAL,
            "STRIDE_COMPRESS"   : STRIDE_COMPRESS,
            "PADDING_STD"       : PADDING_STD,

            # 3. Regularización y Normalización
            "DROPOUT_ENCODER"   : DROPOUT_ENCODER,
            "DROPOUT_DECODER"   : DROPOUT_DECODER,
            "ACT_RELU"          : ACT_RELU,
            "ACT_NONE"          : ACT_NONE
        }
    )

    config = wandb.config

    ## instanciamos modelo
    metanca_model = models.MetaNCASegmenter(ae_params=params_encoder, nca_steps=config.max_nca_steps, nca_hidden_dims=config.nca_hidden_dims).to(device)

    ## cargamos pesos del autoencoder preentrenado (Fase 1)
    metanca_model.ae.load_state_dict(torch.load("med_ae_v1_checkpoint_ep30.pth", map_location=device))


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
    optimizer = torch.optim.Adam([
        {'params': [p for p in metanca_model.nca.parameters() if p.requires_grad]},
        {'params': [p for p in metanca_model.param_predictor.parameters() if p.requires_grad]}
    ], lr=1e-4)

    criterion1 = bce_loss
    criterion2 = dice_loss
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
    pool_latente = NCAPool(pool_size=config.pool_size, channels=config.CHANNELS_LATENT, h=80, w=80, device=device, CHANNELS_LEVEL_1=config.CHANNELS_LEVEL_1)
   
    ## rastrear capas convolucionales
    wandb.watch(metanca_model, log="all", log_freq=100)

    print(f"Iniciando Fase 2 - Entrenamiento del MetaNCA Segmenter por {config.epochs} épocas...")

    for epoch in range(config.epochs):
        total_loss = 0.0

        for i, (imgs, masks) in enumerate(prostate_train_loader):
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

            target_masks = target_masks.unsqueeze(1).float()  # Aseguramos que tenga la dimensión de canal
            
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

            ## logeamos en wandb
            wandb.log({
                "batch/total_loss": loss.item(),  # Multiplicamos para reflejar la pérdida real del batch completo
                "batch/loss_cross_entropy": loss_ce.item(),
                "batch/loss_dice": loss_dice.item()
            })


            if (i + 1) % 10 == 0:
                print(f"\n✅ Van {i + 1} batches procesados exitosamente.")
                print(f"📊 Loss promedio actual: {total_loss / (i + 1):.4f}")
                print(f"💧 Leak Factor actual: {metanca_model.nca.leak_factor.item():.4f}")


        ## -- Fase  de evaluación en Test Set al final de cada época --
        # calculamos promedio epoca
        epoch_train_loss = total_loss / len(prostate_train_loader)
        

        # LOG GLOBAL DE LA ÉPOCA
        wandb.log({
            "epoch/train_loss_promedio": epoch_train_loss,
            "epoch": epoch + 1
        })  

        print(f"\n🎯 Época [{epoch+1}/{config.epochs}] - Loss promedio: {total_loss / len(prostate_train_loader):.4f}")
        # Guardar checkpoint local de la Fase 2
        torch.save(metanca_model.state_dict(), f"med_metanca_v1_checkpoint_ep{epoch+1}.pth")
        torch.cuda.empty_cache()

    wandb.finish()  # Finalizamos la sesión de W&B

if __name__ == "__main__":
    main()
            
            


