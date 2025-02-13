#!/usr/bin/env python3

import sys
import uuid
from owlready2 import *
import rdflib
from rdflib.namespace import RDF, RDFS, OWL, XSD

# Path to your Protege ontology file
ontology_path = './multifactor.rdf'

# Load the ontology
g = rdflib.Graph()
g.parse(ontology_path, format="xml")

# Define namespaces
EX = rdflib.Namespace("http://example.org/multifactor#")
g.bind("ex", EX)

phones_class = EX.Phones
subclasses = set()

# Find direct subclasses of phones_class
for s, p, o in g.triples((None, RDFS.subClassOf, phones_class)):
    subclasses.add(s)
    
individuals = set()
for subclass in subclasses:
    for s, p, o in g.triples((None, RDF.type, subclass)):
        individuals.add(s)

# Load the ontology using owlready2
onto = get_ontology(ontology_path).load()

# Create classes and data properties dynamically
with onto:
    for individual in individuals:
        individual_name = individual.split("#")[1]
        
        # Create a class for the individual
        individual_class = type(individual_name, (Thing,), {})
        for s, p, o in g.triples((individual, None, None)):
            if isinstance(o, rdflib.Literal):
                property_name = p.split("#")[1]
                property_value = o.toPython()
                
                # Determine the type of the property value
                if isinstance(property_value, str):
                    property_type = str
                elif isinstance(property_value, bool):
                    property_type = bool
                elif isinstance(property_value, int):
                    property_type = int
                else:
                    continue  # Skip unsupported types
                
                # Create a data property for the individual
                data_property_class = type(property_name, (DataProperty,), {})
                data_property_class.domain = [individual_class]
                data_property_class.range = [property_type]
                
                # Assign the value to the individual
                setattr(individual_class, property_name, property_value)

    # Add SWRL rules to the ontology
    new_rule = Imp()
    attacker_class = type("attacker", (DataProperty,), {})
    attacker_class.range = [bool]
    new_rule.set_as_rule(f"""id_5b95bc47726e4ec58c1d58e25e3c0f45__18(?u) ^ screenLength(?u, ?l) ^ notEqual(?l, 93) -> attacker(?u, true)""")
# Run the reasoner
sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True)