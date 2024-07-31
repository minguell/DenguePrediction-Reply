import json
import os

def collect_regions_from_file(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    regions = set()
    
    for item in data['data']:
        region = item.get('region')
        if region:
            regions.add(region['name'])
    
    return regions

def collect_regions_from_directory(directory):
    all_regions = set()
    
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            regions = collect_regions_from_file(file_path)
            all_regions.update(regions)
    
    return list(all_regions)

def write_regions_to_file(regions, output_file):
    with open(output_file, 'w') as file:
        for region in regions:
            file.write(f'{region}\n')

# Example usage
directory = os.path.dirname(os.path.abspath(__file__))  # Directory of the current script
output_file = 'regions.txt'
regions = collect_regions_from_directory(directory)
write_regions_to_file(regions, output_file)
print(f'Regions have been written to {output_file}')