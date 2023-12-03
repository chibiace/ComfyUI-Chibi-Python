# ComfyUI-Chibi-Python.py

import json
import requests
import random
import math
import time
import sys

# --------------------------- 
#   Edit Values Below
# ---------------------------

# ComfyUI server and port
server = "http://127.0.0.1:8188"

# if random_seed is True the seed will be random, otherwise uses fixed_seed
random_seed = True
fixed_seed = 8008135

# Checkpoint file inside the ComfyUI Install/models/checkpoints directory
# revcounter_v10LCM.safetensors https://civitai.com/models/72564?modelVersionId=241937
checkpoint = "revcounter_v10LCM.safetensors"

# Prompts
positive_prompt = "beautiful scenery nature glass bottle landscape,purple galaxy bottle"
negative_prompt = "nsfw,nude,text,watermark"

# Image size and number of images
height = 512
width = 512
batch_size = 1

# lcm sampler settings
sampler = "lcm"
steps = 6
cfg = 1.8

# eg. euler sampler
# sampler = "euler"
# steps = 20
# cfg = 7

# ---------------------------


def seed():
    if random_seed:
        return math.floor(random.random() * 10000000000000000)
    else:
        return fixed_seed

# modified api workflow
prompt_text = {
  "3": {
    "inputs": {
      "seed": seed(),
      "steps": steps,
      "cfg": cfg,
      "sampler_name": sampler,
      "scheduler": "normal",
      "denoise": 1,
      "model": [
        "4",
        0
      ],
      "positive": [
        "6",
        0
      ],
      "negative": [
        "7",
        0
      ],
      "latent_image": [
        "5",
        0
      ]
    },
    "class_type": "KSampler"
  },
  "4": {
    "inputs": {
      "ckpt_name": checkpoint
    },
    "class_type": "CheckpointLoaderSimple"
  },
  "5": {
    "inputs": {
      "width": height,
      "height": width,
      "batch_size": batch_size
    },
    "class_type": "EmptyLatentImage"
  },
  "6": {
    "inputs": {
      "text": positive_prompt,
      "clip": [
        "4",
        1
      ]
    },
    "class_type": "CLIPTextEncode"
  },
  "7": {
    "inputs": {
      "text": negative_prompt,
      "clip": [
        "4",
        1
      ]
    },
    "class_type": "CLIPTextEncode"
  },
  "8": {
    "inputs": {
      "samples": [
        "3",
        0
      ],
      "vae": [
        "4",
        2
      ]
    },
    "class_type": "VAEDecode"
  },
  "9": {
    "inputs": {
      "filename_prefix": "ComfyUI",
      "images": [
        "8",
        0
      ]
    },
    "class_type": "SaveImage"
  }
}



def download_file(filename):
    request = requests.get(server+f"/view?filename={filename}")
    if request.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(request.content)
    print(f'Saved {filename}')

def send_prompt():
    request_data = json.dumps({"prompt": prompt_text}).encode('utf-8')
    request = requests.post(server+"/prompt",data=request_data)
    return json.loads(request.content)

response = send_prompt()
try:
    prompt_id = response["prompt_id"]
except:
    print("Error:")
    print(json.dumps(response,indent=2))
    sys.exit()

while True:
    request = requests.get(server+f"/history/{prompt_id}")
    response = json.loads(request.content)
    if len(response) > 0:
        if response[prompt_id]["outputs"]:
            for i in response[prompt_id]["outputs"]["9"]["images"]:
                filename = i["filename"]
                download_file(filename)
            break
        else:
            print("something went wrong")
            break
    time.sleep(1)

# Have a nice day :)

