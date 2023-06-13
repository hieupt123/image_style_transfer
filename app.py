### 1. Imports and class names setup ### 
import gradio as gr
import os
import torch
import utils
from typing import Tuple, Dict
from model import TransformerNet
from torchvision import transforms
from PIL import Image

# Get model
model_dir = '/models'
models = list(Path(model_dir).glob("*/*.pth.tar"))
models = sorted(models)

# Get style image
style_dir = '/style-images'
style_list = list(Path(style_dir).glob("*"))
style_list = sorted(style_list)

# Get examples
example_list = [["examples/" + example] for example in os.listdir("examples")]

def transfer(image, model):
    device = 'cpu'

    width = image.size[0]
    height = image.size[1]

    if width > 750 or height > 500:
      iamge = image.thumbnail((712, 474))

    # load model
    style_model = TransformerNet()
    state_dict = torch.load(models[int(model)], map_location=torch.device('cpu'))
    style_model.load_state_dict(state_dict["state_dict"])

    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(image)
    content_image = content_image.unsqueeze(0).to(device)

    style_model.eval()
    with torch.no_grad():
        style_model.to(device)
        output = style_model(content_image).cpu()

    img = utils.deprocess(output[0])
    img = Image.fromarray(img)
    return img, style_list[int(model)]

# Create title, description and article strings
title = "Image Style Transfer"
description = "Choose a image that you want to transfer and the corresponding style. The app will be transfer your image. You will have received new image."
article = "Model have created base on paper [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/pdf/1603.08155v1.pdf)."

image_output_1 = gr.Image(label='Tranfer') # output result
image_output_2 = gr.Image(label='Style Image') # Show style image

# Create the Gradio demo
demo = gr.Interface(fn=transfer, # mapping function from input to output
                    inputs=[gr.Image(type="pil", label='Input'),
                            gr.Dropdown(choices=[i.parent.name for i in models], value='rain_princess', type='index', label="Style", info="Chooses kind of style image")], # what are the inputs?
                    outputs=[image_output_1, image_output_2], # our fn has two outputs, therefore we have two outputs
                    label = ['One', "Two"],
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article)

# Launch the demo!
demo.launch()
