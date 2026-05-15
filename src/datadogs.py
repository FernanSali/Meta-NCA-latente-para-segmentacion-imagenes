import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn as nn 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


## transformaciones para el dataset a trabaar

def adjust_mask(x):
    return (x-1).squeeze().long()

target_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.PILToTensor(),
    transforms.Lambda(adjust_mask) # Convierte {1,2,3} -> {0,1,2}
])

img_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


## dataset de perros y gatos

pet_data_train = datasets.OxfordIIITPet(root="data", split="trainval", target_types="segmentation" ,download=True, transform=img_transform, target_transform=target_transform)

pet_data_test = datasets.OxfordIIITPet(root="data", split="test", target_types="segmentation" ,download=True, transform=img_transform, target_transform=target_transform)


## dataloader
batch_size = 4 # Ajusta según tu memoria de video (VRAM)

train_loader = DataLoader(
    pet_data_train, 
    batch_size=batch_size, 
    shuffle=True,      
    #num_workers=2,     # Acelera la carga de datos
    #pin_memory=True    # Mejora velocidad de transferencia a GPU
)


test_loader = DataLoader(
    pet_data_test, 
    batch_size=batch_size, 
    shuffle=True,      
    #num_workers=2,     # Acelera la carga de datos
    #pin_memory=True    # Mejora velocidad de transferencia a GPU
)

## Funcion para probar modelo IA y ver resultados
def probar_modelo(model, test_loader, device, num_images=3):
    model.eval()
    images, masks = next(iter(test_loader)) # Tomamos un batch del test_loader

    criterion = nn.CrossEntropyLoss() 
    
    # Movemos al device
    images = images.to(device)
    masks = masks.to(device)
    
    with torch.no_grad(): # Desactiva el cálculo de gradientes (ahorra memoria)
        # Tu modelo devuelve (reconstruction, latent)
        logits, _ = model(images)
        
        # Los logits tienen forma (Batch, 3, H, W). 
        # Aplicamos argmax en la dimensión de canales (1) para obtener la clase [0, 1, 2]
        preds = torch.argmax(logits, dim=1)

        loss = criterion(logits, masks).item()
        print(f"Loss en el batch de prueba: {loss:.4f}")

    # Visualización
    fig, axes = plt.subplots(num_images, 3, figsize=(12, num_images * 4))
    
    for i in range(num_images):
        # 1. Imagen Original (revertir normalización si la usaste)
        img_vis = images[i].cpu().permute(1, 2, 0).numpy()
        axes[i, 0].imshow(img_vis)
        axes[i, 0].set_title("Imagen Original")
        axes[i, 0].axis("off")
        
        # 2. Máscara Real (Ground Truth)
        axes[i, 1].imshow(masks[i].cpu().numpy(), cmap='viridis')
        axes[i, 1].set_title("Máscara Real")
        axes[i, 1].axis("off")
        
        # 3. Predicción del Modelo
        axes[i, 2].imshow(preds[i].cpu().numpy(), cmap='viridis')
        axes[i, 2].set_title("Predicción AI")
        axes[i, 2].axis("off")


    plt.tight_layout()
    plt.show()


## Funcion para ver el nca, ve la pare encoding decoding sin el nca y con el nca implementado
def comparar_modelos(model1, model2, test_loader, device, num_images=3):
    # 1. Definir el criterio de pérdida (debe ser el mismo usado en el entrenamiento)
    criterion = nn.CrossEntropyLoss()
    
    model1.eval()
    model2.eval()
    
    # Tomamos un batch del test_loader
    images, masks = next(iter(test_loader)) 
    
    # Movemos al device
    images = images.to(device)
    masks = masks.to(device)
    
    with torch.no_grad(): 
        # Inferencia de ambos modelos
        # Recordar que ambos devuelven (reconstruction, latent)
        logits1, _ = model1(images)
        logits2, _ = model2(images)

        # 2. Calcular la pérdida específica de este batch para cada modelo
        loss1 = criterion(logits1, masks).item()
        loss2 = criterion(logits2, masks).item()

        # Obtener las clases [0, 1, 2] con argmax
        preds1 = torch.argmax(logits1, dim=1)
        preds2 = torch.argmax(logits2, dim=1)

    # Visualización
    fig, axes = plt.subplots(num_images, 4, figsize=(18, num_images * 4))
    
    # Añadimos un título general con las pérdidas promedio del batch
    fig.suptitle(f'Comparación de Desempeño\nLoss AE Base: {loss1:.4f} | Loss con NCA: {loss2:.4f}', 
                 fontsize=16, fontweight='bold')

    for i in range(num_images):
        # 1. Imagen Original
        img_vis = images[i].cpu().permute(1, 2, 0).numpy()
        axes[i, 0].imshow(img_vis)
        axes[i, 0].set_title("Imagen Original")
        axes[i, 0].axis("off")
        
        # 2. Máscara Real (Ground Truth)
        axes[i, 1].imshow(masks[i].cpu().numpy(), cmap='viridis')
        axes[i, 1].set_title("Máscara Real")
        axes[i, 1].axis("off")
        
        # 3. Predicción Encoder-Decoder
        axes[i, 2].imshow(preds1[i].cpu().numpy(), cmap='viridis')
        axes[i, 2].set_title(f"AE Base")
        axes[i, 2].axis("off")

        # 4. Predicción con NCA 
        axes[i, 3].imshow(preds2[i].cpu().numpy(), cmap='viridis')
        axes[i, 3].set_title(f"AE + NCA")
        axes[i, 3].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajuste para que no se solape el suptitle
    plt.show()


def calculo_perdida(model, test_loader, device):
    # --- Cálculo de Loss en Test Set ---
    model.eval()
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            
            # Obtenemos logits del modelo (ignoramos el espacio latente _)
            logits, _ = model(images)
            
            loss = criterion(logits, masks)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"\n[EVAL] Loss promedio en Test Loader: {avg_test_loss:.4f}")

if __name__ == "__main__":

    # Verificar que los datos se cargan correctamente
    for images, masks in train_loader:
        print(f"Batch de imágenes: {images.shape}")  # Debería ser [batch_size, 3, 256, 256]
        print(f"Batch de máscaras: {masks.shape}")   # Debería ser [batch_size, 256, 256]
        break  # Solo verificar el primer batch


    # Configuración de hardware acelerado
    device = (
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )
    print(f"Usando el dispositivo: {device}")

    ## instaciamos el modelo
    params = {
    'in_c': 3,
    # --- ENCODER ---ig
    'Conv2DParams1': {'out_c': 64, 'kernel_size': 3, 'strides': 1, 'padding': 1, 
                      'activation': 'relu', 'batch_normalization': True, 'dropout_rate': 0.1},
    'Conv2DParams2': {'out_c': 128, 'kernel_size': 3, 'strides': 2, 'padding': 1, 
                      'activation': 'relu', 'batch_normalization': True, 'dropout_rate': 0.1},
    
    # Pass-through debe terminar con 128 canales y stride total de 2 para sumar con Conv2DParams2
    'PassThroughParams1': {'out_c': 64, 'kernel_size': 1, 'strides': 1, 'padding': 0, 
                           'activation': 'relu', 'batch_normalization': True, 'dropout_rate': 0.0},
    'PassThroughParams2': {'out_c': 128, 'kernel_size': 3, 'strides': 2, 'padding': 1, 
                           'activation': 'relu', 'batch_normalization': True, 'dropout_rate': 0.0},
    
    'Conv2DParams3': {'out_c': 256, 'kernel_size': 3, 'strides': 2, 'padding': 1, 
                      'activation': 'relu', 'batch_normalization': True, 'dropout_rate': 0.2},

    # --- DECODER ---
    'Conv2DTransposeParams3': {'out_c': 128, 'kernel_size': 3, 'strides': 2, 'padding': 1, 
                               'activation': 'relu', 'batch_normalization': True, 'dropout_rate': 0.0},
    'Conv2DTransposeParams2': {'out_c': 64, 'kernel_size': 3, 'strides': 2, 'padding': 1, 
                               'activation': 'relu', 'batch_normalization': True, 'dropout_rate': 0.0},
    
    # MixParams recibe d2_out (64) + c1_out (64), mantiene 64 canales
    'MixParams': {'out_c': 64, 'kernel_size': 3, 'strides': 1, 'padding': 1, 
                  'activation': 'relu', 'batch_normalization': True, 'dropout_rate': 0.0},
    
    # Salida final: 3 canales para las 3 clases del dataset
    'Conv2DTransposeParams1': {'out_c': 3, 'kernel_size': 3, 'strides': 1, 'padding': 1, 
                               'activation': None, 'batch_normalization': False, 'dropout_rate': 0.0}
    }

    from model import NCASegmenter

    nca_entero = NCASegmenter(ae_params=params, nca_steps=16).to(device)

    nca_entero.load_state_dict(torch.load("nca_entero4.pth", map_location=device))

    #calculo_perdida(nca_entero, test_loader, device)

    comparar_modelos(nca_entero.ae, nca_entero, test_loader, device, num_images=3)