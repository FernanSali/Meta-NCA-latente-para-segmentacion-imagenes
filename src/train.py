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


## archivo con ejemplos de entrenamientos para las 2 fases del modelo

## funcion referencia para entrenar autoencoder 
## antes hay que:
## ae_model = AutoEncoderDown3(ae_params).to(device)
def train_ae(ae_model, train_loader, epochs=10, device='cuda'):
    '''Entrenemiento del autoencoder solo, fase 1'''
    criterion = nn.CrossEntropyLoss() # Ideal para las 3 clases del trimapa
    optimizer = optim.Adam(ae_model.parameters(), lr=1e-4)

    for epoch in range(epochs):  # Número de épocas
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device) # Debe ser (N, H, W) con valores {0, 1, 2}

            # Forward
            # 'reconstruction' será tu predicción de máscara (N, 3, H, W)
            outputs, _ = ae_model(images) 
            
            loss = criterion(outputs, masks)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        print(f"Época [{epoch+1}/{10}]")
        torch.cuda.empty_cache()

    # torch.save(ae_model.state_dict(), "")


## Pool de entrenamiento para estabilidad de los NCA
class NCAPool:
    '''pool de estados latentes para la estabilidad de los NCA'''
    def __init__(self, pool_size, channels, h, w, device):
        # El pool guarda el estado latente completo (N, C, H, W) 
        self.size = pool_size
        self.pool = torch.zeros(pool_size, channels, h, w).to(device)
        self.device = device

    def sample(self, batch_size, current_latent):
        '''muestrea elemetos del pool de forma estocastica (95% viejos, 5% nuevos/reset)'''
        ## Seleccionamos índices al azar del pool
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)

        ## copiamos lo estados guardados en esos indices
        sampled_states = self.pool[idx].clone()

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

        return x_input, idx

    def update(self, idx, new_states):
        # Guardamos los estados evolucionados de vuelta en el buffer, sin gradietes
        self.pool[idx] = new_states.detach()


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
    ], lr=1e-3)

    criterion = nn.CrossEntropyLoss()
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


    total_loss = 0

    ## inicializamos el pool latnete
    pool_latente = NCAPool(pool_size=1024, channels=256, h=64, w=64, device=device)
    print(" Iniciando loop de prueba preliminar blindado...")

    for epoch in range(epochs):
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
            
            # Generamos los pesos dinámicos a partir del lote actual
            latent_vector = latent_base.mean(dim=[2, 3])
            dynamic_weights = metanca_model.param_predictor(latent_vector)
            
            # MUESTREO DEL POOL
            # En lugar de usar siempre 'latent_base', dejamos que el pool decida estocásticamente
            latent_input, pool_indices = pool_latente.sample(imgs.shape[0], latent_base)
            
            # El NCA evoluciona el estado seleccionado (ya sea inicial o intermedio del pool)
            latent_evolved = metanca_model.nca(latent_input, weights=dynamic_weights, steps=metanca_model.nca_steps)
            
            # ACTUALIZACIÓN DEL POOL
            # Guardamos los estados resultantes en el pool para la siguiente oportunidad
            pool_latente.update(pool_indices, latent_evolved)
            
            # DECODER FINAL Y PÉRDIDA
            # El decoder toma los estados evolucionados para proyectar a la máscara final
            d3_out = metanca_model.ae.trans_conv_3(latent_evolved)
            d2_out = metanca_model.ae.trans_conv_2(d3_out)
            sum_dec = d2_out + c1_out # Inyección del skip connection original
            mixed = metanca_model.ae.mix_layer(sum_dec)
            outputs = metanca_model.ae.trans_conv_1(mixed)
            
            ## optimizacion estandar
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            # Restricción matemática de estabilidad del autómata celular
            with torch.no_grad():
                metanca_model.nca.leak_factor.clamp_(1e-3, 1e3)
            
            total_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(f"\n✅ Van {i + 1} batches procesados exitosamente.")
                print(f"📊 Loss promedio actual: {total_loss / (i + 1):.4f}")
                print(f"💧 Leak Factor actual: {metanca_model.nca.leak_factor.item():.4f}")
            
            
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