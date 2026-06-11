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

# HIPERPARÁMETROS ARQUITECTÓNICOS

CHANNELS_IN       = 3   # Imágenes RGB de entrada
CHANNELS_LEVEL_1  = 64  # Resolución alta (Skip Connection c1_out / d2_out)
CHANNELS_LEVEL_2  = 128  # Resolución intermedia (Salida de compresión y Pass-Through)
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
def evaluar_en_test(model, test_loader, criterion1, criterion2):
    """Calcula las métricas de validación en datos no vistos."""
    model.eval()
    test_loss_ce = 0.0
    test_loss_dice = 0.0
    test_loss_total = 0.0
    
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device).long()
            
            outputs, latent_space = model(images)
            
            loss_ce = criterion1(outputs, masks)
            loss_dice = criterion2(outputs, masks)
            
            test_loss_ce += loss_ce.item()
            test_loss_dice += loss_dice.item()

            test_loss_total += (loss_ce + loss_dice).item()
            
    num_batches = len(test_loader)
    return (
        test_loss_total / num_batches,
        test_loss_ce / num_batches,
        test_loss_dice / num_batches,
    )


import torch.optim as optim
import torch.nn as nn 
import torch.nn.functional as F


from src.datadogs import dog_train_loader, dog_test_loader

from src.train import MulticlassDiceLoss, SpatialContinuityLoss, NCAPool


def main():
    # Inicializamos W&B absorbiendo la metadata estructural en el config
    wandb.init(
        entity="fsalinab-pontificia-universidad-cat-lica-de-chile",
        project="meta-nca-fase2",
        name="metanca_v4",
        config={
            "learning_rate": 1e-4,
            "epochs": 30,
            "batch_size": len(dog_train_loader.dataset) // len(dog_train_loader),
            "alpha_tv": 1e-1,

            "encoder_base_channels": params_encoder2['Conv2DParams1']['out_c'], # dimension skip connection
            "encoder_base_intermediate_channels": params_encoder2['Conv2DParams2']['out_c'],
            "latent_channels": params_encoder2['Conv2DParams3']['out_c'] ,      # canales del Nca (ADN)

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
    metanca_model = models.MetaNCASegmenter(ae_params=params_encoder2, nca_steps=config.max_nca_steps, nca_hidden_dims=config.nca_hidden_dims).to(device)

    ## cargamos pesos del autoencoder preentrenado (Fase 1)
    metanca_model.ae.load_state_dict(torch.load("ae4.1_checkpoint_ep14.pth", map_location=device))

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
    ], lr=config.learning_rate)

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
    pool_latente = NCAPool(pool_size=config.pool_size, channels=config.latent_channels, h=64, w=64, device=device, CHANNELS_LEVEL_1 = config.encoder_base_channels)

    ## rastrear capas convolucionales
    wandb.watch(metanca_model, log="all", log_freq=100)


    print(f"Iniciando Fase 2 - Entrenamiento del MetaNCA Segmenter por {config.epochs} épocas...")

    for epoch in range(config.epochs):
        metanca_model.train()  # Activa modo entrenamiento general para el predictor
        metanca_model.ae.eval() # Fuerza al Autoencoder a mantenerse estático

        total_loss = 0.0

        pasos_acumulacion = 2

        # Inicializamos limpios antes del bucle
        optimizer.zero_grad(set_to_none=True)

        for i, (imgs, masks) in enumerate(dog_train_loader):
            imgs, masks = imgs.to(device), masks.to(device)
            
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
                pasos_tensor = torch.randint(low=config.min_nca_steps, high=config.max_nca_steps//2 + 1, size=(1,))
                steps_nca = pasos_tensor.item()
            else:
                pasos_tensor = torch.randint(low=config.min_nca_steps, high=config.max_nca_steps + 1, size=(1,))
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

            loss = (loss_ce + loss_dice)/pasos_acumulacion #+ loss_tv

            loss.backward()
            # Escudo anti-explosión final
            #torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, metanca_model.parameters()), max_norm=1.0)

            #  Actualización cada 2 batches, para que le de la memoria al compu
            if (i + 1) % pasos_acumulacion == 0 or (i + 1) == len(dog_train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
            
            # Restricción matemática de estabilidad del autómata celular
            with torch.no_grad():
                metanca_model.nca.leak_factor.clamp_(1e-3, 1e-1)
            
            total_loss += loss.item()

            ## logeamos a wandb
            ## logeamos en wandb
            wandb.log({
                "batch/total_loss": loss.item()*pasos_acumulacion,  # Multiplicamos para reflejar la pérdida real del batch completo
                "batch/loss_cross_entropy": loss_ce.item(),
                "batch/loss_dice": loss_dice.item()
            })

        ## -- Fase  de evaluación en Test Set al final de cada época --
        # calculamos promedio epoca
        epoch_train_loss = total_loss / len(dog_train_loader)
        
        # Corremos la métrica sobre el test set (los datos que el modelo jamás vio)
        val_loss, val_ce, val_dice = evaluar_en_test(
            metanca_model, dog_test_loader, criterion1, criterion2
        )

        # LOG GLOBAL DE LA ÉPOCA
        wandb.log({
            "epoch/train_loss_promedio": epoch_train_loss,
            "epoch/test_loss_total": val_loss,
            "epoch/test_loss_ce": val_ce,
            "epoch/test_loss_dice": val_dice,
            "epoch": epoch + 1
        })


        print(f"🎯 Época [{epoch+1}/{config.epochs}] | Train Loss: {epoch_train_loss:.4f} | Test Dice Loss: {val_dice:.4f}")

        # Guardar checkpoint local de la Fase 2
        torch.save(metanca_model.state_dict(), f"metancav4.fase2_checkpoint_ep{epoch+1}.pth")
        torch.cuda.empty_cache()

    wandb.finish()  # Finalizamos la sesión de W&B


if __name__ == "__main__":
    main()