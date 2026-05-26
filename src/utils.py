import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn as nn 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


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



def reporte_memoria():
    # Memoria actualmente usada por tensores
    allocated = torch.cuda.memory_allocated() / 1024**2 
    # Memoria total reservada por PyTorch (el "caché")
    reserved = torch.cuda.memory_reserved() / 1024**2
    
    print(f"Memoria en Tensores: {allocated:.2f} MB")
    print(f"Memoria Reservada (Caché): {reserved:.2f} MB")
    print(f"Memoria 'Libre' dentro del Caché: {(reserved - allocated):.2f} MB")

## pa limpiar 
## torch.cuda.empty_cache()