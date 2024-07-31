import json
import os
import glob

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Initialize counters for male and female mosquitoes
male_count = 0
female_count = 0

# Iterate through all JSON files in the current directory
for filepath in glob.glob(os.path.join(current_directory, '*.json')):
    print(f"Processing file: {filepath}")
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

# Print the results
print(f"Male mosquitoes: {male_count}")
print(f"Female mosquitoes: {female_count}")