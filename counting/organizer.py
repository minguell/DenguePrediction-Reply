import os

# Step 1: Read the MOSQUITO_COUNTS.TXT file
with open('mosquito_counts.txt', 'r') as file:
    lines = file.readlines()

# Step 2: Parse the extracted data
mosquito_counts = {}
for line in lines:
    if '.json' in line:
        filename = line.strip().replace(':', '')
        male_count = int(lines[lines.index(line) + 1].split(': ')[1])
        female_count = int(lines[lines.index(line) + 2].split(': ')[1])
        mosquito_counts[filename] = (male_count, female_count)

# Step 3: Iterate over each JSON file and append the mosquito counts
for filename, counts in mosquito_counts.items():
    if os.path.exists(filename):
        with open(filename, 'a') as json_file:
            json_file.write(f"\nMale mosquitoes: {counts[0]}\nFemale mosquitoes: {counts[1]}\n")