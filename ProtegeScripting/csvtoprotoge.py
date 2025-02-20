import os
import pandas as pd
import rdflib
from rdflib.namespace import RDF, RDFS, OWL, XSD

csv_folder = './100data'
ontology_path = './multifactor.rdf'

g = rdflib.Graph()
if os.path.exists(ontology_path):
    g.parse(ontology_path, format="xml")
else:
    g.bind("ex", "http://example.org/multifactor#")

EX = rdflib.Namespace("http://example.org/multifactor#")
g.bind("ex", EX)

data_properties = [
    "latitude", "longitude", "ipString", "currentTime", "availableMemory", "rssi",
    "timezone", "Processors", "Battery", "Vendor", "Model", "systemPerformance",
    "cpu", "accel", "gyro", "magnet", "screenWidth", "screenLength",
    "screenDensity", "hasTouchScreen", "hasCamera", "hasFrontCamera",
    "hasMicrophone", "hasTemperatureSensor"
]

# Make sure Phones is declared as a class
g.add((EX.Phones, RDF.type, OWL.Class))

# Explicitly declare propertyof and attacker as DatatypeProperties
g.add((EX.propertyof, RDF.type, OWL.DatatypeProperty))
g.add((EX.propertyof, RDFS.domain, EX.Phones))
g.add((EX.propertyof, RDFS.range, XSD.string))

g.add((EX.attacker, RDF.type, OWL.DatatypeProperty))
g.add((EX.attacker, RDFS.domain, EX.Phones))
g.add((EX.attacker, RDFS.range, XSD.boolean))

def guess_xsd_range_from_dtype(dtype):
    if pd.api.types.is_integer_dtype(dtype):
        return XSD.integer
    if pd.api.types.is_float_dtype(dtype):
        return XSD.float
    if pd.api.types.is_bool_dtype(dtype):
        return XSD.boolean
    return XSD.string

# Step 1: Check each CSV to set up data properties and their ranges
for filename in os.listdir(csv_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(csv_folder, filename)
        df = pd.read_csv(file_path)
        for col in data_properties:
            if col in df.columns:
                dp = EX[col]
                g.add((dp, RDF.type, OWL.DatatypeProperty))
                g.add((dp, RDFS.domain, EX.Phones))

                col_dtype = df[col].dtype
                if col_dtype == object:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        if pd.api.types.is_float_dtype(df[col].dtype):
                            if df[col].dropna().eq(df[col].dropna().round()).all():
                                df[col] = df[col].astype('Int64')
                    except:
                        pass

                col_dtype = df[col].dtype
                # If it's an IP, force string
                if col == "ipString":
                    xsd_range = XSD.string
                else:
                    xsd_range = guess_xsd_range_from_dtype(col_dtype)
                # Replace or set the range for this property
                g.set((dp, RDFS.range, xsd_range))

# Step 2: Create individuals and add data property assertions
for filename in os.listdir(csv_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(csv_folder, filename)
        df = pd.read_csv(file_path)

        for col in data_properties:
            if col in df.columns and col != "ipString":
                if df[col].dtype == object:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except:
                        pass
                if pd.api.types.is_float_dtype(df[col]):
                    if df[col].dropna().eq(df[col].dropna().round()).all():
                        df[col] = df[col].astype('Int64')

        # Use the filename (sans extension) to build an individual name
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
                    if prop_range == XSD.integer:
                        val_literal = rdflib.Literal(float(val), datatype=XSD.float)
                    elif prop_range == XSD.float:
                        val_literal = rdflib.Literal(float(val), datatype=XSD.float)
                    elif prop_range == XSD.boolean:
                        val_literal = rdflib.Literal(bool(val), datatype=XSD.boolean)
                    else:
                        val_literal = rdflib.Literal(str(val), datatype=XSD.string)
                    g.add((phone_individual, dp, val_literal))

            # Now use the two data properties we declared:
            g.add((phone_individual, EX.propertyof, rdflib.Literal(username, datatype=XSD.string)))
            g.add((phone_individual, EX.attacker, rdflib.Literal(False, datatype=XSD.boolean)))

# Finally, save the updated graph
g.serialize(destination=ontology_path, format="xml")
print("Ontology updated.")
