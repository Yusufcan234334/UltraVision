import torch
import torch.nn as siniragi
from PIL import Image
from transformers import CLIPTokenizer

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
vocab_size = tokenizer.vocab_size
print(vocab_size)
# CPU kullan
device = torch.device("cpu")

x = torch.load("x.pt")
y = torch.load("y.pt")
SEQ_LEN = x.shape[1]

torch.manual_seed(42)


class muhtisimmodel(siniragi.Module):
    def __init__(self, giris, genislemecikis, katmansayisi, branchsayisi, cikis):
        super().__init__()
        self.embedding = siniragi.Embedding(vocab_size, giris)
        self.genisletici = siniragi.Sequential(
            siniragi.Unflatten(1, (256, 4, 4)),
            siniragi.ConvTranspose2d(256, 128, 4, 2, 1), siniragi.ReLU(),
            siniragi.ConvTranspose2d(128, 64, 4, 2, 1), siniragi.ReLU(),
            siniragi.ConvTranspose2d(64, 32, 4, 2, 1), siniragi.ReLU(),
            siniragi.ConvTranspose2d(32, 16, 4, 2, 1), siniragi.ReLU(),
            siniragi.ConvTranspose2d(16, 8, 4, 2, 1), siniragi.ReLU(),
            siniragi.ConvTranspose2d(8, 4, 4, 2, 1), siniragi.ReLU(),
            siniragi.ConvTranspose2d(4, 3, 4, 2, 1),
        )
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
        self.attention = siniragi.MultiheadAttention(embed_dim=neuroncountoutson, num_heads=heads, batch_first=True)
        self.output1 = siniragi.Linear(neuroncountoutson, cikis)
        self.token_query = siniragi.Parameter(torch.randn(1, 1, giris))
        heads_token = 4
        while giris % heads_token != 0 and heads_token > 1:
            heads_token //= 2
        self.token_attention = siniragi.MultiheadAttention(embed_dim=giris, num_heads=heads_token, batch_first=True)
        self.pos_embedding = siniragi.Embedding(SEQ_LEN, giris)

    def forward(self, x):
        pozisyonlar = torch.arange(x.shape[1], device=x.device)
        x = self.embedding(x) + self.pos_embedding(pozisyonlar)
        q = self.token_query.expand(x.shape[0], -1, -1)
        pooled, _ = self.token_attention(q, x, x)
        x = pooled.squeeze(1)
        ilkx = x
        branchciktilari = []

        for katmanlar in self.branchler:
            x = ilkx

            for katmannum, katman in enumerate(katmanlar):
                x = katman(x)

                if katmannum != len(katmanlar) - 1:
                    x = torch.relu(x)

            branchciktilari.append(x)

        branchciktilari = torch.stack(branchciktilari, dim=1)
        attended, attentionweights = self.attention(branchciktilari, branchciktilari, branchciktilari)
        weights = torch.softmax(self.fusion_weights, dim=0)
        fused = (attended + self.residual_weight * branchciktilari)
        fused = fused * weights.view(1, -1, 1)
        fused = fused.sum(dim=1)
        fused = self.output1(fused)
        fused = self.genisletici(fused)
        return fused


model = muhtisimmodel(64, 128, 4, 4, 4096)

model = model.to(device)

toplam_veri = len(x)
val_size = int(toplam_veri * 0.2)
indices = torch.randperm(toplam_veri)

val_indices = indices[:val_size]

x_val = x[val_indices]
y_val = y[val_indices]

model.load_state_dict(
    torch.load("ultravision-2335.pth", map_location=device)
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