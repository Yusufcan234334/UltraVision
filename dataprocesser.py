import base64
import gc
import os
import time

import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTokenizer
from groq import Groq

DATA_DIR = "datas"
IMG_SIZE = 512
SEQ_LEN = 64
OUT_DIR = "."
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
CHUNK_SIZE = 10
CHUNK_DIR = os.path.join(OUT_DIR, "_chunks")

GROQ_VISION_MODEL = "qwen/qwen3.6-27b"
GROQ_CAPTION_PROMPT = (
    "Describe this image in one short sentence, suitable as a caption for an "
    "image generation model's training data. Be concrete and visual, no preamble."
)

groq_client = Groq()


def caption_with_groq(image_path):
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    mime = "jpeg" if ext == "jpg" else ext

    completion = groq_client.chat.completions.create(
        model=GROQ_VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": GROQ_CAPTION_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}"}},
                ],
            }
        ],
        max_completion_tokens=100,
        temperature=0.3,
        reasoning_effort="none",
    )
    return completion.choices[0].message.content.strip()


def find_pairs(data_dir, auto_caption=True):
    if not os.path.isdir(data_dir):
        raise RuntimeError(f"'{data_dir}' klasörü bulunamadı.")

    files = os.listdir(data_dir)
    stems = {}

    for f in files:
        full = os.path.join(data_dir, f)
        if not os.path.isfile(full):
            continue
        stem, ext = os.path.splitext(f)
        ext = ext.lower()
        if ext in IMAGE_EXTS:
            stems.setdefault(stem, {})["img"] = full
        elif ext == ".txt":
            stems.setdefault(stem, {})["txt"] = full

    pairs = []
    missing_img = []
    need_caption = []
    for stem in sorted(stems.keys()):
        entry = stems[stem]
        img_path = entry.get("img")
        txt_path = entry.get("txt")
        if img_path and txt_path:
            pairs.append((stem, img_path, txt_path))
        elif img_path and not txt_path:
            need_caption.append((stem, img_path))
        elif txt_path and not img_path:
            missing_img.append(stem)

    if need_caption:
        if auto_caption:
            print(f"{len(need_caption)} görselin txt'si yok, Groq ile otomatik açıklama üretiliyor...")
            for stem, img_path in tqdm(need_caption, desc="caption uretiliyor"):
                txt_path = os.path.join(data_dir, stem + ".txt")
                try:
                    caption = caption_with_groq(img_path)
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(caption)
                    pairs.append((stem, img_path, txt_path))
                except Exception as e:
                    print(f"  UYARI: {stem} icin caption uretilemedi, atlandi: {e}")
                time.sleep(0.2)
        else:
            print(f"UYARI: {len(need_caption)} görselin txt'si yok, atlandı.")

    if missing_img:
        print(f"UYARI: {len(missing_img)} txt'nin görseli yok, atlandı: {missing_img[:10]}{' ...' if len(missing_img) > 10 else ''}")

    pairs.sort(key=lambda p: p[0])
    return pairs


def to_tensor_512(img):
    img = img.convert("RGB")
    w, h = img.size
    scale = IMG_SIZE / max(w, h)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
    paste_x = (IMG_SIZE - new_w) // 2
    paste_y = (IMG_SIZE - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))

    arr = torch.frombuffer(bytearray(canvas.tobytes()), dtype=torch.uint8)
    arr = arr.reshape(IMG_SIZE, IMG_SIZE, 3).permute(2, 0, 1).float() / 255.0
    return arr


def main():
    pairs = find_pairs(DATA_DIR)
    if not pairs:
        raise RuntimeError(f"'{DATA_DIR}' içinde eşleşen resim+txt çifti bulunamadı.")

    print(f"{len(pairs)} eşleşen çift bulundu. {CHUNK_SIZE}'ar 'ar parca halinde islenecek.")

    print("Tokenizer indiriliyor...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    os.makedirs(CHUNK_DIR, exist_ok=True)
    chunk_paths = []
    all_used_stems = []

    x_buf, y_buf, stem_buf = [], [], []

    def flush_chunk():
        nonlocal x_buf, y_buf, stem_buf
        if not x_buf:
            return
        chunk_x = torch.stack(x_buf)
        chunk_y = torch.stack(y_buf)
        idx = len(chunk_paths)
        path = os.path.join(CHUNK_DIR, f"chunk_{idx:05d}.pt")
        torch.save({"x": chunk_x, "y": chunk_y, "stems": stem_buf}, path)
        chunk_paths.append(path)
        all_used_stems.extend(stem_buf)
        x_buf.clear()
        y_buf.clear()
        stem_buf = []
        del chunk_x, chunk_y
        gc.collect()

    for stem, img_path, txt_path in tqdm(pairs, desc="isleniyor"):
        try:
            img = Image.open(img_path)
            img_tensor = to_tensor_512(img)
            img.close()

            with open(txt_path, "r", encoding="utf-8") as f:
                desc = f.read().strip()

            tok = tokenizer(
                desc,
                padding="max_length",
                truncation=True,
                max_length=SEQ_LEN,
                return_tensors="pt",
            )["input_ids"][0]

            x_buf.append(tok)
            y_buf.append(img_tensor)
            stem_buf.append(stem)
        except Exception as e:
            print(f"Atlandi ({stem}): {e}")
            continue

        if len(x_buf) >= CHUNK_SIZE:
            flush_chunk()

    flush_chunk()

    if not chunk_paths:
        raise RuntimeError("Hicbir cift islenemedi.")

    print(f"{len(chunk_paths)} parca diske yazildi, x.pt/y.pt olarak birlestiriliyor...")

    x_final = None
    y_final = None
    for path in tqdm(chunk_paths, desc="birlestiriliyor"):
        chunk = torch.load(path)
        if x_final is None:
            x_final, y_final = chunk["x"], chunk["y"]
        else:
            x_final = torch.cat([x_final, chunk["x"]], dim=0)
            y_final = torch.cat([y_final, chunk["y"]], dim=0)
        del chunk
        gc.collect()

    assert x_final.shape[0] == y_final.shape[0]

    torch.save(x_final, os.path.join(OUT_DIR, "x.pt"))
    torch.save(y_final, os.path.join(OUT_DIR, "y.pt"))

    with open(os.path.join(OUT_DIR, "order.txt"), "w", encoding="utf-8") as f:
        for i, stem in enumerate(all_used_stems):
            f.write(f"{i}\t{stem}\n")

    for path in chunk_paths:
        os.remove(path)
    try:
        os.rmdir(CHUNK_DIR)
    except OSError:
        pass

    print(f"Hazir: x.pt {tuple(x_final.shape)}  y.pt {tuple(y_final.shape)}")
    print(f"order.txt yazildi ({len(all_used_stems)} satir).")
    print('Kullanim: x = torch.load("x.pt"); y = torch.load("y.pt")')


if __name__ == "__main__":
    main()