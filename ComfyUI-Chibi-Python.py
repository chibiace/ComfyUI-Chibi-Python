# ComfyUI-Chibi-Python.py

import json
import requests
import random
import math
import time

server = "http://127.0.0.1:8188"
fixed_seed = 8008135
random_seed = True
checkpoint = "revCounter_lcm.safetensors"
positive_prompt = "beautiful scenery nature glass bottle landscape,purple galaxy bottle"
negative_prompt = "nsfw,nude,text,watermark"
height = 512
width = 512
batch_size = 1

def seed():
    if random_seed:
        return math.floor(random.random() * 10000000000000000)
    else:
        return fixed_seed

prompt_text = {
  "3": {
    "inputs": {
      "seed": seed(),
      "steps": 6,
      "cfg": 1.8,
      "sampler_name": "lcm",
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
prompt_id = response["prompt_id"]

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



