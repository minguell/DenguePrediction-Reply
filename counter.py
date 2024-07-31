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

def write_district_mosquitoes_to_file(district_mosquitoes, output_file):
    with open(output_file, 'w') as file:
        for year, months in district_mosquitoes.items():
            for month, districts in months.items():
                file.write(f'{year}-{month:02d}:\n')
                for district, quantity in districts.items():
                    file.write(f'  {district}: {quantity}\n')

def main():
    directory = os.path.dirname(os.path.abspath(__file__))  # Directory of the current script
    output_file = 'district_mosquitoes.txt'
    
    # Create a directory to store the output files
    output_dir = 'monthly_totals'
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all regions
    regions = collect_regions_from_directory(directory)
    
    # Initialize dictionary to store mosquito counts per district per month per year
    district_mosquitoes = {year: {month: {region: 0 for region in regions} for month in range(1, 13)} for year in range(2017, 2024)}
    
    # Loop through years from 2017 to 2023
    for year in range(2017, 2024):
        # Loop through file numbers from 1 to 54
        for i in range(1, 55):
            # Determine the month based on the file number
            if i <= 4:
                month = 1
            elif i <= 8:
                month = 2
            elif i <= 12:
                month = 3
            elif i <= 16:
                month = 4
            elif i <= 20:
                month = 5
            elif i <= 24:
                month = 6
            elif i <= 28:
                month = 7
            elif i <= 32:
                month = 8
            elif i <= 36:
                month = 9
            elif i <= 40:
                month = 10
            elif i <= 44:
                month = 11
            else:
                month = 12

            file_name = f'{year}_{i}.json'
            
            try:
                # Load JSON data from file
                with open(file_name, 'r') as file:
                    data = json.load(file)
                
                # Iterate through each data entry
                for entry in data['data']:
                    # Check if 'inspection' and 'mosquitoes' keys exist and 'inspection' is not None
                    if 'inspection' in entry and entry['inspection'] is not None and 'mosquitoes' in entry['inspection']:
                        region = entry.get('region', {}).get('name')
                        if region:
                            # Iterate through each mosquito and sum up the quantity
                            for mosquito in entry['inspection']['mosquitoes']:
                                district_mosquitoes[year][month][region] += mosquito['pivot']['quantity']
            
            except FileNotFoundError: 
                print(f'File {file_name} not found.')
            except json.JSONDecodeError:
                print(f'Error decoding JSON in file {file_name}.')
            except KeyError as e:
                print(f'Missing key {e} in file {file_name}.')
            except TypeError as e:
                print(f'Type error: {e} in file {file_name}.')

    # Write the results to a file
    write_district_mosquitoes_to_file(district_mosquitoes, output_file)
    print(f'District mosquito counts have been written to {output_file}')

if __name__ == "__main__":
    main()