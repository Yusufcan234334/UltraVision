import torch
import torch.nn as siniragi
from torch.utils.data import Dataset, DataLoader, TensorDataset

x = torch.load("x.pt")
y = torch.load("y.pt")
result = torch.zeros((x.shape[0], 512), dtype=x.dtype)

n = min(x.shape[1], 512)
result[:, :n] = x[:, :n]

x = result
torch.manual_seed(42)


class muhtisimmodel(siniragi.Module):
    def __init__(self, giris, genislemecikis, katmansayisi, branchsayisi, cikis):
        super().__init__()

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
        return fused

model = muhtisimmodel(512, 1024, 2, 3,4096)
print("Model parametre sayısı:")
print(sum(p.numel() for p in model.parameters()))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
tahmin = model(x)
losshesaplayici = siniragi.BCEWithLogitsLoss()
loss = losshesaplayici(tahmin, y)
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
dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=16, shuffle=True ) #,generator=generator


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
        tahminler = (torch.sigmoid(tahmin) >= 0.5).float()

        tp = ((tahminler == 1) & (y_val == 1)).sum().float()
        fp = ((tahminler == 1) & (y_val == 0)).sum().float()
        fn = ((tahminler == 0) & (y_val == 1)).sum().float()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        print(f"Precision: {precision.item():.3f}  Recall: {recall.item():.3f}  F1: {f1.item():.3f}")
        print(f"TP: {int(tp.item())}  FP: {int(fp.item())}  FN: {int(fn.item())}")
    model.train()

if __name__ == "__main__":
    train(model,loader, )