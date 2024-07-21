"""
in order to run this do the following 

If you are not using Linux, do NOT proceed, see instructions for macOS and Windows.

Clone this repository and navigate to LLaVA folder
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
Install Package
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
Install additional packages for training cases
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
Upgrade to latest code base
git pull
pip install -e .

# if you see some import errors when you upgrade,
# please try running the command below (without #)
# pip install flash-attn --no-build-isolation --no-cache-dir

then place this file under /LLaVA/llava/serve/caption.py

and run it with this command please adjust the path to your dataset file:

python -m llava.serve.caption     --model-path liuhaotian/llava-v1.6-34b     --data-file /home/bas06400/Thesis/CV_training_set.txt     --load-8bit --device cuda:0

"""
 



import argparse
import torch
from PIL import Image
import cv2
from io import BytesIO
import requests
import os
import re
import sys
import json

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from transformers import TextStreamer

def load_image_from_frame(video_path, frame_number):
    #print(video_path)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = cap.read()
    if success:
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    else:
        return None

def extract_class_info(filename):
    # Extract the base name of the file
    base_name = filename.split('/')[-1]
    
    # Use a regular expression to find the part before the first underscore followed by a number
    match = re.search(r'^(.+?)_\d', base_name)
    if match:
        class_info = match.group(1)
        # Replace all underscores in the class name with spaces
        class_info = class_info.replace('_', ' ')
        return class_info
    else:
        return "Unknown Class"  # Default case if the pattern is not found

def process_video(video_path, args, model, tokenizer, image_processor):
    conv = conv_templates[args.conv_mode].copy() 
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #print(total_frames)
    frames_to_capture = [total_frames // 3, total_frames // 2, 2 * total_frames // 3]
    descriptions = []
    class_info = extract_class_info(video_path)

    for frame_number in frames_to_capture:
        #print(f"processing frame: {frame_number}")
        image = load_image_from_frame(video_path, frame_number)
        if image is None:
            print(f"No valid image at frame {frame_number}")
            continue
        image_size = image.size
        image_tensor = process_images([image], image_processor, model.config).to(args.device, dtype=torch.float16)

        for i in range(3):
            #print(f"This is attempt{i+1}")
            inp = f"Describe the human action in this frame in just one sentence. " #related to {class_info}
            if model.config.mm_use_im_start_end:
                
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:

                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            #print(prompt)
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(args.device)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            with torch.inference_mode():
                #print("Input IDs shape:", input_ids.shape)
                #print("Image tensor shape:", image_tensor.shape)
                #print("Image size:", image_size)
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image_size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    #streamer=streamer,
                    use_cache=False)
                #print("Generation done")
                conv = conv_templates[args.conv_mode].copy()  
            description = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            if description == '':
                description = class_info
            descriptions.append(description)
    
    return descriptions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, default="descriptionstestdaa0val.jsonl")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    args = parser.parse_args()

    # Model and tokenization
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)
    

    
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles
    
    with open(args.data_file, 'r') as file:
        lines = file.readlines()
    
    total_lines = len(lines)
    results = []
    save_interval = 100

    for index, line in enumerate(lines):
        video_path = re.split(r'(?<=[iy]) | (?=n)', line)[0]
        clip_id = video_path.split('/')[-1].replace('.avi', '') 
        descriptions = process_video(os.path.join('/home/bas06400/daa', video_path), args, model, tokenizer, image_processor)
        results.append({"clip_id": clip_id, "text": descriptions})

        # Save partial results after every save_interval videos
        if (index + 1) % save_interval == 0 or (index + 1) == total_lines:
            with open(args.output_file, 'a') as file:  # Use 'a' mode to append to the file
                for result in results:
                    json.dump(result, file)
                    file.write('\n')
            results = []  # Clear the list after saving

        # Print progress
        progress = (index + 1) / total_lines * 100
        print(f"Processing {index + 1}/{total_lines} ({progress:.2f}%)", end='\r')
        sys.stdout.flush()


    print("\nFinished processing all lines.")
