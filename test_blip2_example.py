import torch
from PIL import Image
import requests

from transformers import Blip2Processor, Blip2ForConditionalGeneration
import random

import pickle
import time
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
#os.environ['HF_HOME'] = '/var/scratch/ybi530/data'
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

print(device)
img_path = '/var/scratch/ybi530/data/data/train2014/'

## generate prompt
df = pd.read_csv('/var/scratch/ybi530/result/intention.csv',index_col=None)
print(df.head(3))
# df

def prompt_intention(row):
    use_columns = ['target_action', 'action1', 'action2',
       'action3', 'action4']
    choices = ''
    for i in use_columns:
        choices += row[i]
    return 'What am I supposed to do '+row['intention']+'? pick two of the best option from the following ones. '+choices

df['prompt_two_option'] = df.apply(prompt_intention, axis=1)
print('Create prompt')


#os.environ['HF_HOME'] = '/var/scratch/ybi530/data'
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xxl",cache_dir = '/var/scratch/ybi530/data/model')
print('finishload processor')

model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xxl", cache_dir = '/var/scratch/ybi530/data/model',device_map="auto")
print('finishload model')

df['result_two_option'] = ''
s1 = 0
s2 = 0
for i in df.index:
    raw_image = Image.open(img_path+df.image_url[i]).convert('RGB')
    question = df['prompt_two_option'][i]

    inputs = processor(raw_image, question, return_tensors="pt").to(device)
    # print('input generate')
    out = model.generate(**inputs)
    result = processor.decode(out[0], skip_special_tokens=True)
    df['result_two_option'][i] = result
    # print(i,result,question)
    if i>s1+50:
        print(i)
        s1 = s1+50
    if i>s2+500:
        df.to_csv('temp.csv')
        s2 = s2+500

df.to_csv('temp.csv')
    
   


## generate prompt run
