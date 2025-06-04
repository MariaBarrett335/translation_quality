import fireducks.pandas as pd
import numpy as np
import os

"""
To run this again, create enwiki-20220201-clean.txt as described here https://github.com/GermanT5/wikipedia2corpus
In this file, each new article is separated by a blank line.
It parses the file and adds an article id to each and saves them in multiple parquet files
"""

# Use a more memory-efficient approach: process and save in batches
chunk_size = 100000  # Reduce chunk size to handle memory constraints
reader = pd.read_csv('enwiki-20220201-clean.txt', 
                    sep='\t', 
                    header=None,
                    chunksize=chunk_size,
                    dtype={0: 'string'},  # Use string dtype to save memory
                    on_bad_lines='skip')  # Skip problematic lines

# Track group IDs across chunks
group_id_offset = 0
current_group = 0
batch_num = 0
batch_size = 10  # Number of chunks to process before saving

for chunk_num, chunk in enumerate(reader):
    try:
        # Name the column
        chunk.columns = ['sentence']
        
        # Use a more memory-efficient way to identify blank rows
        chunk['is_blank'] = chunk['sentence'].isna() | (chunk['sentence'] == '')
        
        # Calculate group IDs for this chunk
        chunk['group_id'] = chunk['is_blank'].cumsum() + group_id_offset
        
        # Shift the group_id
        chunk['group_id'] = chunk['group_id'].shift(1, fill_value=current_group)
        
        # Filter out blank rows
        chunk_grouped = chunk[~chunk['is_blank']].copy()
        chunk_grouped = chunk_grouped[['group_id', 'sentence']]
        
        # Update offsets for next chunk
        last_group = chunk['group_id'].max() if not chunk.empty else current_group
        current_group = last_group
        group_id_offset = current_group
        
        # Save the processed chunk directly to a file
        batch_file = f'data/wiki_batch_{batch_num}.parquet'
        chunk_grouped.to_parquet(batch_file, index=False)
        
        print(f"Processed chunk {chunk_num}, current group ID: {current_group}, saved to {batch_file}")
        
        # Increment batch number every batch_size chunks
        if (chunk_num + 1) % batch_size == 0:
            batch_num += 1
            
    except Exception as e:
        print(f"Error processing chunk {chunk_num}: {e}")
        continue  # Skip problematic chunks

print(f"Finished processing chunks. Results saved to batch files.")
