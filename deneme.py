import torch
import torch.nn as siniragi
from PIL import Image

# CPU kullan
device = torch.device("cpu")

x = torch.load("x.pt")
y = torch.load("y.pt")
x = x.float()

torch.manual_seed(42)


class muhtisimmodel(siniragi.Module):
    def __init__(self, giris, genislemecikis, katmansayisi, branchsayisi, cikis):
        super().__init__()
        self.genisletici = siniragi.Sequential(siniragi.ReLU(),siniragi.Linear(4096, 4096), siniragi.ReLU(),siniragi.Unflatten(1, (256, 4, 4)),siniragi.ConvTranspose2d(256, 128, 4, 2, 1), siniragi.ReLU(),siniragi.ConvTranspose2d(128, 64, 4, 2, 1), siniragi.ReLU(),siniragi.ConvTranspose2d(64, 32, 4, 2, 1), siniragi.ReLU(),siniragi.ConvTranspose2d(32, 16, 4, 2, 1), siniragi.ReLU(),siniragi.ConvTranspose2d(16, 8, 4, 2, 1), siniragi.ReLU(),siniragi.ConvTranspose2d(8, 4, 4, 2, 1), siniragi.ReLU(),siniragi.ConvTranspose2d(4, 3, 4, 2, 1))
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

        self.attention = siniragi.MultiheadAttention(
            embed_dim=neuroncountoutson,
            num_heads=heads,
            batch_first=True
        )

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


model = muhtisimmodel(64, 128, 4, 4, 4096)

# Model CPU'da
model = model.to(device)

toplam_veri = len(x)
val_size = int(toplam_veri * 0.2)
indices = torch.randperm(toplam_veri)

val_indices = indices[:val_size]

x_val = x[val_indices]
y_val = y[val_indices]

model.load_state_dict(
    torch.load("ultravision.pth", map_location=device)
)

model.eval()

N = 8
N = min(N, x_val.shape[0])

with torch.no_grad():

    tahmin = model(
        x_val[:N].to(device)
    )

    tahmin = torch.clamp(tahmin, 0, 1)


def tensor_to_pil(t):
    arr = (
        t.permute(1, 2, 0)
        .numpy() * 255
    ).astype("uint8")

    return Image.fromarray(arr)


IMG_SIZE = y.shape[-1]

gap = 8

grid = Image.new(
    "RGB",
    (
        IMG_SIZE * 2 + gap,
        IMG_SIZE * N + gap * (N - 1)
    ),
    (30, 30, 30)
)


for i in range(N):

    gercek = tensor_to_pil(y_val[i])

    tahmin_img = tensor_to_pil(tahmin[i])

    y_off = i * (IMG_SIZE + gap)

    grid.paste(
        gercek,
        (0, y_off)
    )

    grid.paste(
        tahmin_img,
        (IMG_SIZE + gap, y_off)
    )


grid.save("karsilastirma.png")


mse_per_sample = (
    (tahmin - y_val[:N]) ** 2
).mean(dim=(1, 2, 3))


for i in range(N):

    print(
        f"ornek {i}  mse: "
        f"{mse_per_sample[i].item():.5f}"
    )


print(
    "kaydedildi: karsilastirma.png  "
    "(sol: gercek, sag: tahmin)"
)