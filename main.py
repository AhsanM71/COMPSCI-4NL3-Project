from datasets import load_dataset
from datasets import concatenate_datasets
import random
import pandas as pd


def save_groups(groups, paths):
    for i in range(0, 8):
        group_df = pd.DataFrame(groups[i])
        
        new_group_df = pd.DataFrame({
            'id': group_df.index,
            'body': group_df['body'],
            'label': ''
        })
        new_group_df.to_csv(paths[i], index=False)
        
def load_and_sample_data(dataset_name, sample_size, seed=42):
    dataset = load_dataset(dataset_name)
    data = dataset['train']
    return data.shuffle(seed=seed).select(range(sample_size))

def main():
    
    total = 960
    half = int(total/2)
    max = int(total/8)
    
    sampled_data = load_and_sample_data('mjw/stock_market_tweets', total)

    num_duplicates = int(0.15 * len(sampled_data))

    duplicated_data = sampled_data.select(range(num_duplicates)) 
    unique_data = sampled_data.select(range(num_duplicates, len(sampled_data)))

    set_A = duplicated_data.select(range(num_duplicates)) 
    set_B = duplicated_data.select(range(num_duplicates)) 
    
    set_A_group1 = set_A.select(range(int(0.7 * len(set_A))))
    set_A_group2 = set_A.select(range(int(0.7 * len(set_A)), len(set_A))) 

    set_B_group3 = set_B.select(range(int(0.7 * len(set_B)))) 
    set_B_group4 = set_B.select(range(int(0.7 * len(set_B)), len(set_B)))
    
    group1 = unique_data.select(range(0,max-len(set_A_group1)))
    group2 = unique_data.select(range(max-len(set_A_group1), (max-len(set_A_group1) + max-len(set_A_group2))))
    
    group3 = unique_data.select(range((max-len(set_A_group1) + max-len(set_A_group2)), ((max-len(set_A_group1) + max-len(set_A_group2)) + (max-len(set_B_group3)))))
    start = ((max-len(set_A_group1) + max-len(set_A_group2)) + (max-len(set_B_group3)))
    group4 = unique_data.select(range(start,start + (max-len(set_B_group4))))
    
    group5 = sampled_data.select(range(half, half + 120))
    group6 = sampled_data.select(range(half + 120, half + 240))
    group7 = sampled_data.select(range(half + 240, half + 360))
    group8 = sampled_data.select(range(half + 360, half + 480))
    
    group1 = concatenate_datasets([set_A_group1, group1])
    group2 = concatenate_datasets([set_A_group2, group2])
    group3 = concatenate_datasets([set_B_group3, group3])
    group4 = concatenate_datasets([set_B_group4, group4])
    
    groups = [group1, group2, group3, group4, group5, group6, group7, group8]
    filenames = [f'Datasets/data{i}.csv' for i in range(1, 9)]
    save_groups(groups, filenames)

if __name__ == "__main__":
    main()