import json
import os
import glob

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Initialize a dictionary to store counts per file
file_counts = {}

# Iterate through all JSON files in the current directory
for filepath in glob.glob(os.path.join(current_directory, '*.json')):
    print(f"Processing file: {filepath}")
    male_count = 0
    female_count = 0
    with open(filepath, 'r') as file:
        data = json.load(file)
        for item in data['data']:
            inspection = item.get('inspection')
            if inspection and 'mosquitoes' in inspection:
                for mosquito in inspection['mosquitoes']:
                    species_name = mosquito.get('name').lower()
                    gender = mosquito.get('gender')
                    print(f"Found mosquito - Species: {species_name}, Gender: {gender}")
                    if species_name == 'aedes aegypti':
                        if gender == 0:
                            male_count += 1
                        elif gender == 1:
                            female_count += 1
    # Store the counts for the current file
    file_counts[os.path.basename(filepath)] = {'male': male_count, 'female': female_count}

# Write the results to a file
output_file = 'mosquito_counts.txt'
with open(output_file, 'w') as file:
    for filename, counts in file_counts.items():
        file.write(f"{filename}:\n")
        file.write(f"  Male mosquitoes: {counts['male']}\n")
        file.write(f"  Female mosquitoes: {counts['female']}\n")

print(f"Counts have been written to {output_file}") 