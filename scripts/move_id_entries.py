import json
import os

def move_entries():
    # File paths
    old_ids_path = os.path.join('ground_truth', 'raw_llm_responses', 'old_ids', 'results.json')
    new_ids_path = os.path.join('ground_truth', 'raw_llm_responses', 'new_ids', 'results.json')
    
    # Read old IDs file
    with open(old_ids_path, 'r') as f:
        old_ids_data = json.load(f)
    
    # Read new IDs file
    with open(new_ids_path, 'r') as f:
        new_ids_data = json.load(f)
    
    # Separate entries
    new_id_entries = []
    old_id_entries = []
    
    for entry in old_ids_data:
        if entry['id_type'] == 'new':
            new_id_entries.append(entry)
        else:
            old_id_entries.append(entry)
    
    # Add new entries to new IDs file
    new_ids_data.extend(new_id_entries)
    
    # Write back to files
    with open(old_ids_path, 'w') as f:
        json.dump(old_id_entries, f, indent=4)
    
    with open(new_ids_path, 'w') as f:
        json.dump(new_ids_data, f, indent=4)
    
    print(f"Moved {len(new_id_entries)} entries from old IDs to new IDs file")
    print(f"New IDs file now has {len(new_ids_data)} entries")
    print(f"Old IDs file now has {len(old_id_entries)} entries")

if __name__ == '__main__':
    move_entries()
