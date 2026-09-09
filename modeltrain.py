import torch
import torch.nn as siniragi
from torch.utils.data import Dataset, DataLoader, TensorDataset


x = torch.load("x.pt")
y = torch.load("y.pt")
x = x.float()

torch.manual_seed(42)


class muhtisimmodel(siniragi.Module):
    def __init__(self, giris, genislemecikis, katmansayisi, branchsayisi, cikis):
        super().__init__()
        self.genisletici = siniragi.Sequential(siniragi.Linear(4096, 4096), siniragi.ReLU(),siniragi.Linear(4096, 4096), siniragi.ReLU(),siniragi.Unflatten(1, (256, 4, 4)),siniragi.ConvTranspose2d(256, 128, 4, 2, 1), siniragi.ReLU(),siniragi.ConvTranspose2d(128, 64, 4, 2, 1), siniragi.ReLU(),siniragi.ConvTranspose2d(64, 32, 4, 2, 1), siniragi.ReLU(),siniragi.ConvTranspose2d(32, 16, 4, 2, 1), siniragi.ReLU(),siniragi.ConvTranspose2d(16, 8, 4, 2, 1), siniragi.ReLU(),siniragi.ConvTranspose2d(8, 4, 4, 2, 1), siniragi.ReLU(),siniragi.ConvTranspose2d(4, 3, 4, 2, 1))
        self.branchler = siniragi.ModuleList()
        self.fusion_weights = siniragi.Parameter(torch.ones(branchsayisi))
        self.residual_weight = siniragi.Parameter(torch.tensor(1.0))

        for branch in range(branchsayisi):
            katmanlar = siniragi.ModuleList()

            neuroncountin = giris
            neuroncountout = genislemecikis

            for i in range(katmansayisi):
                katmanlar.append(
                    siniragi.Linear(neuroncountin, neuroncountout)
                )

                if i == katmansayisi - 1:
                    break
                else:
                    neuroncountin = neuroncountout
                    neuroncountout = neuroncountout * 2


            for i in range(katmansayisi):
                if i == katmansayisi - 1:
                    neuroncountin = neuroncountout
                    neuroncountout //= 2
                    neuroncountoutson = neuroncountout
                else:
                    neuroncountin = neuroncountout
                    neuroncountout //= 2

                katmanlar.append(
                    siniragi.Linear(neuroncountin, neuroncountout)
                )

            self.branchler.append(katmanlar)
        heads = 4
        while neuroncountoutson % heads != 0 and heads > 1:
            heads //= 2
        self.attention = siniragi.MultiheadAttention(embed_dim=neuroncountoutson,num_heads=heads,batch_first=True)
        self.output1 = siniragi.Linear(neuroncountoutson,cikis)

    def forward(self, x):
        ilkx = x
        branchciktilari = []

        for katmanlar in self.branchler:
            x = ilkx

            for katmannum, katman in enumerate(katmanlar):
                x = katman(x)

                if katmannum != len(katmanlar) - 1:
                    x = torch.relu(x)

            branchciktilari.append(x)

        branchciktilari = torch.stack(branchciktilari,dim=1)
        attended, attentionweights = self.attention(branchciktilari,branchciktilari,branchciktilari)
        weights = torch.softmax(self.fusion_weights,dim=0)
        fused = (attended+ self.residual_weight * branchciktilari)
        fused = fused * weights.view(1, -1, 1)
        fused = fused.sum(dim=1)
        fused = self.output1(fused)
        fused = self.genisletici(fused)
        return fused

model = muhtisimmodel(64, 128, 4, 4,4096)

print("Model parametre sayısı:")
print(sum(p.numel() for p in model.parameters()))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
losshesaplayici = siniragi.MSELoss()
toplam_veri = len(x)
val_size = int(toplam_veri * 0.2)
indices = torch.randperm(toplam_veri)
val_indices = indices[:val_size]
train_indices = indices[val_size:]
x_val = x[val_indices]
y_val = y[val_indices]
x_train = x[train_indices]
y_train = y[train_indices]
print("Train:", x_train.shape, y_train.shape)
print("Validation:", x_val.shape, y_val.shape)

# Sadece train verisinden DataLoader oluştur
dataset = TensorDataset(x_train, y_train)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

def train(model, loader, debug=True):
    losslar = []
    for i in range(149):
        for x_batch, y_batch in loader:

            optimizer.zero_grad()

            tahmin = model(x_batch)

            loss = losshesaplayici(tahmin, y_batch)
            losslar.append(loss.item())

            loss.backward()

            optimizer.step()
        tamlosslar = sum(losslar) / len(losslar)
        losslar = []
        if debug == True: print(f"Epoch {i} ortalama loss: {tamlosslar}")
        wakywakyitstimeforval(model, x_val, y_val)
    torch.save(model.state_dict(), "level1model.pth")

def wakywakyitstimeforval(model, x_val, y_val):
    model.eval()
    with torch.no_grad():
        tahmin = model(x_val)
        mse = losshesaplayici(tahmin, y_val)
        print(f"Val MSE: {mse.item():.5f}")
    model.train()

if __name__ == "__main__":
    train(model,loader, )