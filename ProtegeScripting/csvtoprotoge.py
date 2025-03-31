import os
import pandas as pd
import rdflib
from rdflib.namespace import RDF, RDFS, OWL, XSD
#Phones and (suspicious some xsd:float[>= 8.0f])
csv_folder = './100data'
ontology_path = './multifactor.rdf'

# Load or initialize RDF graph
g = rdflib.Graph()
if os.path.exists(ontology_path):
    g.parse(ontology_path, format="xml")
else:
    g.bind("ex", "http://example.org/multifactor#")

# Define and bind namespace
EX = rdflib.Namespace("http://example.org/multifactor#")
g.bind("ex", EX)

# Data properties to track
data_properties = [
    "latitude", "longitude", "ipString", "currentTime", "availableMemory", "rssi",
    "timezone", "Processors", "Battery", "Vendor", "Model", "systemPerformance",
    "cpu", "accel", "gyro", "magnet", "screenWidth", "screenLength",
    "screenDensity", "hasTouchScreen", "hasCamera", "hasFrontCamera",
    "hasMicrophone", "hasTemperatureSensor"
]

# Define Phones class and essential properties
g.add((EX.Phones, RDF.type, OWL.Class))
g.add((EX.Attackers, RDF.type, OWL.Class))
g.add((EX.propertyof, RDF.type, OWL.DatatypeProperty))
g.add((EX.propertyof, RDFS.domain, EX.Phones))
g.add((EX.propertyof, RDFS.range, XSD.string))

g.add((EX.attacker, RDF.type, OWL.DatatypeProperty))
g.add((EX.attacker, RDFS.domain, EX.Phones))
g.add((EX.attacker, RDFS.range, XSD.boolean))

g.add((EX.suspicious, RDF.type, OWL.DatatypeProperty))
g.add((EX.suspicious, RDFS.domain, EX.Phones))
g.add((EX.suspicious, RDFS.range, XSD.float))

# Determine XSD range based on column and dtype
def guess_xsd_range_from_dtype(dtype, colname):
    if colname == "ipString":
        return XSD.string
    if colname.startswith("has") or pd.api.types.is_bool_dtype(dtype):
        return XSD.boolean
    if pd.api.types.is_numeric_dtype(dtype):
        return XSD.float
    return XSD.string

# Step 1: Define data properties and their ranges
for filename in os.listdir(csv_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(csv_folder, filename)
        df = pd.read_csv(file_path)

        for col in data_properties:
            if col in df.columns:
                dp = EX[col]
                g.add((dp, RDF.type, OWL.DatatypeProperty))
                g.add((dp, RDFS.domain, EX.Phones))

                try:
                    if col.startswith("has"):
                        df[col] = df[col].astype(str).str.lower().map({'true': True, 'false': False})
                    elif col != "ipString":
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
                except:
                    pass

                xsd_range = guess_xsd_range_from_dtype(df[col].dtype, col)
                g.set((dp, RDFS.range, xsd_range))

# Step 2: Add individuals and data property values
for filename in os.listdir(csv_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(csv_folder, filename)
        df = pd.read_csv(file_path)

        for col in data_properties:
            if col in df.columns:
                try:
                    if col.startswith("has"):
                        df[col] = df[col].astype(str).str.lower().map({'true': True, 'false': False})
                    elif col != "ipString":
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
                except:
                    pass

        individual_name = os.path.splitext(filename)[0]
        individual_name = individual_name.replace("-", "")
        individual_name = individual_name.replace(".", "__")
        individual_name = "id_" + individual_name

        username = individual_name.split("__")[0]
        user_class = EX[username]
        g.add((user_class, RDF.type, OWL.Class))
        g.add((user_class, RDFS.subClassOf, EX.Phones))

        phone_individual = EX[individual_name]
        g.add((phone_individual, RDF.type, user_class))

        if not df.empty:
            row = df.iloc[0]
            for col, val in row.items():
                if col in data_properties and not pd.isna(val):
                    dp = EX[col]
                    prop_range = g.value(dp, RDFS.range)

                    # Correct literal creation
                    if prop_range == XSD.float:
                        val_literal = rdflib.Literal(float(val), datatype=XSD.float)
                    elif prop_range == XSD.boolean:
                        val_literal = rdflib.Literal(bool(val), datatype=XSD.boolean)
                    else:
                        val_literal = rdflib.Literal(str(val), datatype=XSD.string)

                    g.add((phone_individual, dp, val_literal))

            # Add general properties
            g.add((phone_individual, EX.propertyof, rdflib.Literal(username, datatype=XSD.string)))
            g.add((phone_individual, EX.attacker, rdflib.Literal(False, datatype=XSD.boolean)))
            g.add((phone_individual, EX.suspicious, rdflib.Literal(0.0, datatype=XSD.float)))

# Save the ontology
g.serialize(destination=ontology_path, format="xml")
print("Ontology updated.")
