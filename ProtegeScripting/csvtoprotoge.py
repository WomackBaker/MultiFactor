import os
import pandas as pd
import rdflib
from rdflib.namespace import RDF, RDFS, OWL, XSD

# Path to the folder containing CSV files
csv_folder = './100data'

# Path to your Protege ontology file
ontology_path = './multifactor.rdf'

# Load or create the ontology
g = rdflib.Graph()
if os.path.exists(ontology_path):
    g.parse(ontology_path, format="xml")
else:
    # Create a new ontology
    g.bind("ex", "http://example.org/multifactor#")
    print("Created new ontology file.")

# Define namespaces
EX = rdflib.Namespace("http://example.org/multifactor#")
g.bind("ex", EX)

# Define a function to create data properties and set their ranges
def create_data_properties(properties, sample_row):
    for prop in properties:
        data_property = EX[prop]
        g.add((data_property, RDF.type, OWL.DatatypeProperty))
        g.add((data_property, RDFS.domain, EX.Phones))
        
        # Infer the range based on the sample value
        sample_value = sample_row[prop]
        if pd.isna(sample_value):
            g.add((data_property, RDFS.range, XSD.string))  # Default to string if sample value is NaN
        elif isinstance(sample_value, int):
            g.add((data_property, RDFS.range, XSD.integer))
        elif isinstance(sample_value, float):
            g.add((data_property, RDFS.range, XSD.float))
        elif isinstance(sample_value, bool):
            g.add((data_property, RDFS.range, XSD.boolean))
        else:
            g.add((data_property, RDFS.range, XSD.string))
    g.add((EX["propertyof"],RDFS.range, XSD.string))
    g.add((EX["attacker"],RDFS.range, XSD.boolean))


# List of data properties
data_properties = [
    "latitude", "longitude", "ipString", "currentTime", "availableMemory", "rssi", "timezone",
    "Processors", "Battery", "Vendor", "Model", "systemPerformance", "cpu", "accel", "gyro",
    "magnet", "screenWidth", "screenLength", "screenDensity", "hasTouchScreen", "hasCamera",
    "hasFrontCamera", "hasMicrophone", "hasTemperatureSensor"
]

# Iterate over all CSV files in the folder to get a sample row for type inference
sample_row = None
for filename in os.listdir(csv_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(csv_folder, filename)
        df = pd.read_csv(file_path)
        sample_row = df.iloc[0]
        break  # Use the first row of the first CSV file as the sample

# Create data properties in the ontology using the sample row for type inference
create_data_properties(data_properties, sample_row)

users = []

# Iterate over all CSV files in the folder
for filename in os.listdir(csv_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(csv_folder, filename)
        df = pd.read_csv(file_path)
        # Use the filename (without extension) as the individual name
        individual_name = os.path.splitext(filename)[0]
        individual_name = individual_name.replace("-", "")
        individual_name = individual_name.replace(".", "__")
        individual_name = "id_"+individual_name
        
        username = individual_name.split("__")[0]
        user = EX[username]
        print(user)
        
        g.add((user, RDF.type, OWL.Class))
        g.add((user, RDFS.subClassOf, EX.Phones))
        phone_individual = EX[individual_name]
        g.add((phone_individual, RDF.type, user))

        '''
        if user not in users:
            users.append(user)
            print("User added")
        else:
            print(f"ADDING {individual_name} SAME AS {username}.1")
            g.add((EX[f"{username}.1"],OWL.sameAs, phone_individual))
        '''
        
        # Assign values to data properties for the first row only
        for _, row in df.iterrows():
            for col, value in row.items():
                if col in data_properties:
                    data_property = EX[col]
                    # Skip NaN values
                    if pd.isna(value):
                        continue
                    # Convert the value to the appropriate data type
                    if g.value(data_property, RDFS.range) == XSD.integer:
                        value = rdflib.Literal(int(value), datatype=XSD.integer)
                    elif g.value(data_property, RDFS.range) == XSD.float:
                        value = rdflib.Literal(float(value), datatype=XSD.float)
                    elif g.value(data_property, RDFS.range) == XSD.boolean:
                        value = rdflib.Literal(bool(value), datatype=XSD.boolean)
                    else:
                        value = rdflib.Literal(str(value), datatype=XSD.string)
                    g.add((phone_individual, data_property, value))
            break  # Only process the first row
        g.add((phone_individual, EX["propertyof"], rdflib.Literal(username, datatype=XSD.string)))
# Save the updated ontology in RDF format
g.serialize(destination=ontology_path, format="xml")

print("Ontology updated.")