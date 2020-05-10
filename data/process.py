import random
import json
from sklearn.model_selection import train_test_split

import argparse
import json
from os.path import exists
from tqdm import tqdm
from underthesea import sent_tokenize
import random

with open('zalo_v2.0.json','r',encoding='utf-8') as json_file:
    data = json.load(json_file)
data2=[]
for item in data:
    if item['text']!='':
        data2.append(item)
# print(data[0])
# 1/0
valid = []
train_new = []
index = list(range(0,len(data2)))
random.shuffle(index)
# print((index))
train_index = index[:int(len(index)*0.8)]
valid_index = index[int(len(index)*0.8)+1:]
# print(valid_index)
for x in valid_index:
    valid.append(data2[x])

for x in train_index:
    train_new.append(data2[x])
print(train_new[0])

with open('train_eng.json','w',encoding='utf-8') as file:
    json.dump(train_new,file)
with open('valid_eng.json','w',encoding='utf-8') as file:
    json.dump(valid,file)


def get_word_count(text):
    # Split by space & remove empty text
    texts = text.split(' ')
    try:
        text_len = len(texts.remove(""))
    except ValueError:
        text_len = len(texts)
    except TypeError:
        return 0

    return text_len

def convert_mode_veryshort(input_file, output_file, encoding):
    with open(input_file, 'r', encoding=encoding) as stream:
        squad = json.load(stream)

    convertedData = []

    # Remove _ symbol in title
    for data in squad['data']:
        data['title'] = " ".join(data['title'].split('_'))

    # Format 2: Sentence as Text
    for data in tqdm(squad['data']):
        for paragraph in data['paragraphs']:
            # Get paragraph split by sentences & determine its start index for easier processing
            para_context = sent_tokenize(paragraph['context'])  # Context split into list of sentences
            para_sent_startidxs = [paragraph['context'].index(sentence) for sentence in para_context]

            # Process question-answer pairs
            for qas in paragraph['qas']:
                # Prepare data to save
                zaloQAS = {
                    'id': qas['id'],
                    'question': qas['question'],
                    'title': data['title'],
                    'label': False if qas['is_impossible'] else True
                }
                _question_len = get_word_count(qas['question'])

                # Loop & get answer text for each qa pair
                if len(qas['answers']) != 0 and qas['is_impossible'] is False \
                        and qas['answers'][0]['answer_start'] != -1:
                    # Only 1 answer, but rephrased
                    answer = qas['answers'][0]

                    # Find the sentence & sentence index that contains the answer
                    _text = None
                    for idx in range(len(para_context)):
                        if para_sent_startidxs[idx] > answer['answer_start']:
                            continue
                        elif para_sent_startidxs[idx] < answer['answer_start'] \
                                < para_sent_startidxs[idx] + len(para_context[idx]):
                            _text = para_context[idx]
                            break
                        else:
                            break
                    zaloQAS['text'] = "" if _text is None else _text
                else:
                    zaloQAS['text'] = para_context[random.randint(0, len(para_context)) - 1] if len(para_context) >= 1 \
                        else ""

                # Add data instance
                convertedData.append(zaloQAS)

    # Export converted data
    with open(output_file, 'w', encoding=encoding) as stream:
        stream.write(json.dumps(convertedData, ensure_ascii=False))

# convert_mode_veryshort('train-v2.0.json','zalo_v2.0.json','utf-8')
# count=0
# count2=0
# with open('zalo_v2.0.json','r',encoding='utf-8') as file:
#     data=json.load(file)
# for item in data:
#     if item['text']!='':
#         count+=1
#     if item['label']!=True:
#         count2+=1
#         print(item)
#         2/0
# print(count,count2)