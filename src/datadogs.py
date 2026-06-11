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

dog_train_loader = DataLoader(
    pet_data_train, 
    batch_size=batch_size, 
    shuffle=True,      
    #num_workers=2,     # Acelera la carga de datos
    #pin_memory=True    # Mejora velocidad de transferencia a GPU
)


dog_test_loader = DataLoader(
    pet_data_test, 
    batch_size=batch_size, 
    shuffle=True,      
    #num_workers=2,     # Acelera la carga de datos
    #pin_memory=True    # Mejora velocidad de transferencia a GPU
)



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
    for images, masks in dog_train_loader:
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

    comparar_modelos(nca_entero.ae, nca_entero, dog_test_loader, device, num_images=3)