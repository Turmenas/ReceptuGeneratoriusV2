import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import gc
import os
from sklearn.model_selection import train_test_split

# --- NUSTATYMAI ---
INPUT_DIM = 2000     # Turi sutapti su prepare_ml_data.py
EMBEDDING_DIM = 64
BATCH_SIZE = 128
EPOCHS = 50          # Nustatome daug, bet Early Stopping sustabdys anksčiau
PATIENCE = 5         # Kiek epochų laukti, jei rezultatas negerėja

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Mokymui naudojama: {device}")

# --- MODELIO ARCHITEKTŪRA ---
class RecipeNet(nn.Module):
    def __init__(self):
        super(RecipeNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, EMBEDDING_DIM)
        )
        self.decoder = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, INPUT_DIM),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# --- EARLY STOPPING KLASĖ ---
class EarlyStopping:
    def __init__(self, patience=3, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            print(f'   ⚠️ Validacijos rezultatas negerėja ({self.counter}/{self.patience})')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        '''Išsaugo geriausią modelį'''
        torch.save(model.state_dict(), 'best_recipe_model.pth')
        print("   ✅ Aptiktas pagerėjimas! Modelis išsaugotas.")

# --- 1. DUOMENŲ PARUOŠIMAS ---
print("1. Kraunami duomenys ir dalinami (Train/Val)...")

# Krauname duomenis
data = np.load('train_matrix.npz')
matrix = data['matrix'].astype(np.float32)

# Daliname į 80% Train ir 20% Validation
X_train, X_val = train_test_split(matrix, test_size=0.2, random_state=42)

# Konvertuojame į Tensorius
train_tensor = torch.from_numpy(X_train)
val_tensor = torch.from_numpy(X_val)

# Sukuriame loaderius
train_loader = DataLoader(TensorDataset(train_tensor, train_tensor), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(val_tensor, val_tensor), batch_size=BATCH_SIZE, shuffle=False)

# Išvalome RAM (nebereikia originalios matricos)
del matrix, data, X_train, X_val
gc.collect()

# --- 2. MOKYMAS ---
model = RecipeNet().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
early_stopper = EarlyStopping(patience=PATIENCE)

print("2. Pradedamas mokymas su Early Stopping...")

for epoch in range(EPOCHS):
    # --- MOKYMO FAZĖ ---
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        inputs = batch[0].to(device)
        optimizer.zero_grad()
        _, decoded = model(inputs)
        loss = criterion(decoded, inputs)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)

    # --- VALIDACIJOS FAZĖ ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch[0].to(device)
            _, decoded = model(inputs)
            loss = criterion(decoded, inputs)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epocha {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Tikriname ar stabdyti
    early_stopper(avg_val_loss, model)
    
    if early_stopper.early_stop:
        print("🛑 Ankstyvas sustabdymas! Modelis nustojo mokytis.")
        break

# --- 3. UŽBAIGIMAS ---
print("3. Užkraunamas geriausias išsaugotas modelis...")
# Užkrauname geriausius svorius, kuriuos išsaugojo EarlyStopping
model.load_state_dict(torch.load('best_recipe_model.pth'))

# Pervardiname galutiniam naudojimui (kad atitiktų Yummy.py)
torch.save(model.state_dict(), "recipe_model.pth")
if os.path.exists("best_recipe_model.pth"):
    os.remove("best_recipe_model.pth") # Ištriname laikiną failą

# Išvalome atmintį
del train_tensor, val_tensor, train_loader, val_loader
gc.collect()

# --- 4. VEKTORIŲ GENERAVIMAS ---
print("4. Generuojami paieškos vektoriai (su geriausiu modeliu)...")

search_data = np.load('search_matrix.npz')['matrix'].astype(np.float32)
search_tensor = torch.from_numpy(search_data)
search_loader = DataLoader(TensorDataset(search_tensor), batch_size=512, shuffle=False)

all_embeddings = []
model.eval()

with torch.no_grad():
    for batch in search_loader:
        batch_in = batch[0].to(device)
        encoded = model.encoder(batch_in)
        all_embeddings.append(encoded.cpu())

final_embeddings = torch.cat(all_embeddings)

# Normalizacija
norms = final_embeddings.norm(p=2, dim=1, keepdim=True)
normalized_embeddings = final_embeddings / norms.clamp(min=1e-8)

torch.save(normalized_embeddings, 'search_embeddings.pt')
print("✅ Baigta! Vektoriai sugeneruoti.")