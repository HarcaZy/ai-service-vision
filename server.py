from flask import Flask, request, jsonify
import torch
import clip
from PIL import Image

app = Flask(__name__)

device = "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

LABELS = [
    "t-shirt","hoodie","jacket","coat",
    "jeans","shorts","sweatpants",
    "dress","shoes","sneakers","boots"
]

text = clip.tokenize(LABELS).to(device)

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]

    path = "/tmp/img.jpg"
    file.save(path)

    image = preprocess(Image.open(path)).unsqueeze(0).to(device)

    with torch.no_grad():
        img = model.encode_image(image)
        txt = model.encode_text(text)

        logits = (img @ txt.T).softmax(dim=-1)
        label = LABELS[logits.argmax().item()]

    return jsonify({"label": label})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
