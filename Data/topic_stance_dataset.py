import json
import os
import pandas as pd

if __name__ == "__main__":
    final_data = []
    
    # Create output directory if it doesn't exist
    os.makedirs('./Data', exist_ok=True)
    
    for i in range(19):
        json_name = f'./Data/output/topic_stance_dataset_{i}.json'
        with open(json_name, 'r', encoding='utf-8') as file:
            # Parse the JSON data into Python objects
            data = json.load(file)
            
            # If each file contains a list, extend the final list
            if isinstance(data, list):
                final_data.extend(data)
            else:
                # If each file contains a single object, append it
                final_data.append(data)

    # Save the combined data as JSON
    with open('./Data/topic_stance_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)
    
    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(final_data)
    df.to_csv('./Data/topic_stance_dataset.csv', index=False, encoding='utf-8')
        
    print(f"Successfully combined 19 files into JSON and CSV formats in the './Data/' directory")
