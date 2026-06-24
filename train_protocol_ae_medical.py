import torch

from src import model as models

## importamos los dataloaders
from src.datadogs import dog_train_loader, dog_test_loader

from src.utils import probar_modelo, comparar_modelos, calculo_perdida

from src.datadogs import dog_train_loader, dog_test_loader

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



def main():

    # Inicializamos W&B absorbiendo la metadata estructural en el config
    wandb.init(
        entity="fsalinab-pontificia-universidad-cat-lica-de-chile",
        project="medical_ae",
        name="ae_medical_v1",
        config={
            "learning_rate": 1e-4,
            "epochs": 100,
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

    ## instanciamos el autoencoder
    ae_model = models.AutoEncoderDown3(params_encoder).to(device)

    ## entrenamiento ae

    ae_model.to(device)


    criterion1 = bce_loss
    criterion2 = dice_loss

    optimizer = torch.optim.Adam(ae_model.parameters(), lr=config.learning_rate)

    ## rastrear capas convolucionales
    wandb.watch(ae_model, log="all", log_freq=100)

    print(f"Iniciando Fase 1 directamente en el Autoencoder por {config.epochs} épocas...")

    for epoch in range(config.epochs):  # Número de épocas
        ae_model.train()  # Modo entrenamiento

        running_loss = 0.0


        for images, masks in prostate_train_loader:
            images = images.to(device)
            masks = masks.to(device).long() # Debe ser (N, H, W) con valores {0, 1, 2}

            # Forward
            # 'reconstruction' será tu predicción de máscara (N, 3, H, W)
            outputs, latent_space = ae_model(images) 

            # Suma de las 3 losses: Píxel (CE) + Región (Dice) + Suavizado (TV)
            loss_ce = criterion1(outputs, masks.float())
            loss_dice = criterion2(outputs, masks)

            loss = loss_ce + loss_dice
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            ## neuronas muertas 
            num_ceros = (latent_space == 0).sum().float().item()
            porcentaje_neuronas_muertas = num_ceros / latent_space.numel()

                    ## logeamos en wandb
            wandb.log({
                "batch/total_loss": loss.item(),
                "batch/loss_cross_entropy": loss_ce.item(),
                "batch/loss_dice": loss_dice.item(),
                "latent/sparsity": porcentaje_neuronas_muertas
            })

        ## -- Fase  de evaluación en Test Set al final de cada época --
        # calculamos promedio epoca
        epoch_train_loss = running_loss / len(prostate_train_loader)

        ## ACA VER SI AGREGAMOS REVISAR EN EL TEST SET


        #  LOG GLOBAL DE LA ÉPOCA
        wandb.log({
            "epoch/train_loss_promedio": epoch_train_loss,
            "epoch": epoch + 1
        })


        # calculamos promedio epoca
        print(f"Época [{epoch+1}/{config.epochs} - Loss promedio: {epoch_train_loss:.4f}]")

        # Guardar checkpoint local de la Fase 1
        torch.save(ae_model.state_dict(), f"med_ae_v1_checkpoint_ep{epoch+1}.pth")
        torch.cuda.empty_cache()

    wandb.finish()  # Finalizamos la sesión de W&B


if __name__ == "__main__":
    main()