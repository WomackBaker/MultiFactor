import os
import pandas as pd
from owlready2 import *

# Path to the folder containing CSV files
csv_folder = './samplecsv'

# Path to your Protege ontology file
ontology_path = './multifactor.owx'

# Check if the ontology file exists
if os.path.exists(ontology_path):
    # Load the existing ontology
    onto = get_ontology(ontology_path).load()
else:
    print("Ontology file not found.")
    exit()

with onto:
    class CSVIndividual(Thing):
        pass
    
# Define a function to create OWL individuals from CSV data
def create_individual_from_row(row):
    with onto:
        # Create an individual for each row in the CSV
        individual = CSVIndividual()
        for column in row.index:
            # Assuming each column corresponds to a property in the ontology
            prop = getattr(onto, column, None)
            if prop:
                setattr(individual, prop, row[column])
    return individual

# Iterate over all CSV files in the folder
for filename in os.listdir(csv_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(csv_folder, filename)
        df = pd.read_csv(file_path)
        
        # Create OWL individuals from each row in the DataFrame
        for index, row in df.iterrows():
            create_individual_from_row(row)

# Save the updated ontology in OWL/XML format
onto.save(file=ontology_path, format="rdfxml")

# Run the reasoner
with onto:
    sync_reasoner()

print("Ontology updated and reasoner executed.")
