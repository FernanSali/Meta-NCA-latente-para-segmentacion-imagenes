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


# =========================================================================
# 🎛️ DEFINICIÓN DE VARIABLES COMUNES (HIPERPARÁMETROS ARQUITECTÓNICOS)
# =========================================================================

# 1. Configuración de Canales (Flujo Simétrico de Dimensiones)
CHANNELS_IN       = 3   # Imágenes RGB de entrada
CHANNELS_LEVEL_1  = 32  # Resolución alta (Skip Connection c1_out / d2_out)
CHANNELS_LEVEL_2  = 64  # Resolución intermedia (Salida de compresión y Pass-Through)
CHANNELS_LATENT   = 32  # Cuello de botella latente (Espacio del ADN / Entrada del NCA)
CHANNELS_OUT      = 3   # Clases del Trimapa (Fondo, Interior, Contorno)

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

# =========================================================================
# 📐 DICCIONARIO DE PARÁMETROS CONFIGURADO CON VARIABLES COMUNES
# =========================================================================

params_encoder2 = {
    'in_c': CHANNELS_IN,
    
    # ─── ENCODER ──────────────────────────────────────────────────────────
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

## funcion pa evaluar modelo ae en test set
def evaluar_en_test(model, test_loader, criterion1, criterion2, criterion3):
    """Calcula las métricas de validación en datos no vistos."""
    model.eval()
    test_loss_ce = 0.0
    test_loss_dice = 0.0
    test_loss_tv = 0.0
    test_loss_total = 0.0
    
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device).long()
            
            outputs, latent_space = model(images)
            
            loss_ce = criterion1(outputs, masks)
            loss_dice = criterion2(outputs, masks)
            loss_tv = criterion3(outputs)
            
            test_loss_ce += loss_ce.item()
            test_loss_dice += loss_dice.item()
            test_loss_tv += loss_tv.item()
            test_loss_total += (loss_ce + loss_dice + loss_tv).item()
            
    num_batches = len(test_loader)
    return (
        test_loss_total / num_batches,
        test_loss_ce / num_batches,
        test_loss_dice / num_batches,
        test_loss_tv / num_batches
    )

import torch.optim as optim
import torch.nn as nn 
import torch.nn.functional as F

from src.datadogs import dog_train_loader, dog_test_loader

ae_model = models.AutoEncoderDown3(params_encoder2).to(device)
#ae_model.load_state_dict(torch.load("ae_model_og_2.pth", map_location=device))


## vamoe a entrenar el auto-encoder primero
from src.train import MulticlassDiceLoss, SpatialContinuityLoss



def main():
    # Inicializamos W&B absorbiendo la metadata estructural en el config
    wandb.init(
        entity="fsalinab-pontificia-universidad-cat-lica-de-chile",
        project="meta-nca-fase1-autoencoder",
        name="ae_pure_baseline_v2",
        config={
            "learning_rate": 1e-4,
            "epochs": 10,
            "batch_size": len(dog_train_loader.dataset) // len(dog_train_loader),
            "alpha_tv": 1e-1,
            "encoder_base_channels": params_encoder2['Conv2DParams1']['out_c'], # dimension skip connection
            "latent_channels": params_encoder2['Conv2DParams3']['out_c']       # canales del Nca (ADN)
        }
    )

    config = wandb.config

    ## --- Entrenamiento del Autoencoder ---

    ## instanciamos el modelo
    ae_model = models.AutoEncoderDown3(params_encoder2).to(device)
    
    criterion1 = nn.CrossEntropyLoss() # Ideal para las 3 clases del trimapa
    criterion2 = MulticlassDiceLoss() # Para mejorar la segmentación de bordes y detalles finos
    criterion3 = SpatialContinuityLoss(alpha=config.alpha_tv) # Para fomentar la continuidad espacial en las predicciones

    optimizer = optim.Adam(ae_model.parameters(), lr=config.learning_rate)

    ## rastrear capas convolucionales
    wandb.watch(ae_model, log="all", log_freq=100)

    print(f"Iniciando Fase 1 directamente en el Autoencoder por {config.epochs} épocas...")

    for epoch in range(config.epochs):  # Número de épocas
        ae_model.train()  # Modo entrenamiento

        running_loss = 0.0


        for images, masks in dog_train_loader:
            images = images.to(device)
            masks = masks.to(device).long() # Debe ser (N, H, W) con valores {0, 1, 2}

            # Forward
            # 'reconstruction' será tu predicción de máscara (N, 3, H, W)
            outputs, latent_space = ae_model(images) 

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

            ## neuronas muertas 
            num_ceros = (latent_space == 0).sum().float().item()
            porcentaje_neuronas_muertas = num_ceros / latent_space.numel()

            ## logeamos en wandb
            wandb.log({
                "batch/total_loss": loss.item(),
                "batch/loss_cross_entropy": loss_ce.item(),
                "batch/loss_dice": loss_dice.item(),
                "batch/loss_spatial_tv": loss_tv.item(),
                "latent/sparsity": porcentaje_neuronas_muertas
            })

        ## -- Fase  de evaluación en Test Set al final de cada época --
        # calculamos promedio epoca
        epoch_train_loss = running_loss / len(dog_train_loader)
        
        # Corremos la métrica sobre el test set (los datos que el modelo jamás vio)
        val_loss, val_ce, val_dice, val_tv = evaluar_en_test(
            ae_model, dog_test_loader, criterion1, criterion2, criterion3
        )
        
        # 🚀 LOG GLOBAL DE LA ÉPOCA
        wandb.log({
            "epoch/train_loss_promedio": epoch_train_loss,
            "epoch/test_loss_total": val_loss,
            "epoch/test_loss_ce": val_ce,
            "epoch/test_loss_dice": val_dice,
            "epoch/test_loss_tv": val_tv,
            "epoch": epoch + 1
        })

        print(f"🎯 Época [{epoch+1}/{config.epochs}] | Train Loss: {epoch_train_loss:.4f} | Test Dice Loss: {val_dice:.4f}")
        
        # Guardar checkpoint local de la Fase 1
        torch.save(ae_model.state_dict(), f"ae_fase2.1_checkpoint_ep{epoch+1}.pth")
        torch.cuda.empty_cache()
    ## --- Fin entrenamiento autoencoder ---

    wandb.finish()  # Finalizamos la sesión de W&B

if __name__ == "__main__":
    main()