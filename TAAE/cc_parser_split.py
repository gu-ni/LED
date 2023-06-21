import os
import json
from tqdm import tqdm

datadir = 'data'
ccdir = 'data/ccsplit'
os.makedirs(datadir, exist_ok=True)
os.makedirs(ccdir, exist_ok=True)

def parse_commonsense_conversation(source, target):
    source_file = os.path.join(datadir, source)
    target_file = os.path.join(ccdir, target)
    print(f'parsing {source_file}')
    with open(source_file, "r") as input_file:
        with open(target_file, "w") as output_file:
            for line in tqdm(input_file):
                sample = json.loads(line.strip())

                post = sample["post"]
                response = sample["response"]
                
                post_str = " ".join(post)
                response_str = " ".join(response)
                
                output_file.write(post_str + "\n")
                output_file.write(response_str + "\n")

if __name__ == "__main__":
    parse_commonsense_conversation(source='testset.txt', target='test.txt')
    parse_commonsense_conversation(source='trainset.txt', target='train.txt')
    parse_commonsense_conversation(source='validset.txt', target='valid.txt')
