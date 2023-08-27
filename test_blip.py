import sys

import torch
import argparse
import bla_utils as utils

import numpy as np
from tqdm import tqdm

# sys.path.append("/Users/xinyichen/Desktop/Thesis/Experiments0411/BLIP")
from models.blip_itm import blip_itm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


image_size = 384
# image = utlis.load_demo_image(image_size=image_size,device=device)

# caption = 'a woman sitting on the beach with a dog'

# print('text: %s' %caption)

# itm_output = model(image,caption,match_head='itm')
# itm_score = torch.nn.functional.softmax(itm_output,dim=1)[:,1]
# print('The image and text is matched with a probability of %.4f'%itm_score)

# itc_score = model(image,caption,match_head='itc')
# print('The image feature and text feature has a cosine similarity of %.4f'%itc_score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path',type = str,\
        default='/home/xchen/datasets/BLA/original/active_passive_captions_gruen_strict.json', \
        help = "path of caption input")

    args = parser.parse_args()
    items = utils.read_json_file(args.file_path)
    
    # Load BLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth'
    model = blip_itm(pretrained=model_url, image_size=image_size, vit='base', 
                 med_config = "/home/xchen/BLIP/configs/med_config.json"
                 )
    model.eval()
    model = model.to(device=device)
    
    caption_types = ['True1', 'True2', 'False1', 'False2']
    acc = 0
    
    ranks = []
    for item in tqdm(items):
        image_id = item['image_id']
        captions = item['caption_group'][0]
        img_path =  "/home/xchen/datasets/BLA/images/" + str(image_id) + ".jpg"
        image =  utils.load_demo_image(image_size=image_size,device=device, img_path = img_path)
        itc_scores = []
        for i, type in enumerate(caption_types):
            itc_score = model(image,captions[type], match_head='itc')
            itc_scores.append(itc_score.item())
        
        itc_scores = np.array(itc_scores)
        sorted_rank = np.argsort(-1 * itc_scores).argsort() + 1
        ranks.append(sorted_rank)
    
    print(utils.get_rank_statistics(ranks))
        