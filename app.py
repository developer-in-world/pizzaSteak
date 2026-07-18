import gradio as gr
from PIL import Image

from src.predict import predict


def classify(image: Image.Image):
    label, confidence = predict(image)

    return {
        label.capitalize(): confidence,
        ("Steak" if label.lower() == "pizza" else "Pizza"): 1 - confidence
    }


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="🍕 Pizza vs Steak Classifier",
    description="""
Upload a pizza or burger image.

Built using PyTorch + Gradio.
"""
)

if __name__ == "__main__":
    demo.launch()