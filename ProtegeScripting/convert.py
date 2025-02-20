#!/usr/bin/env python3

import sys
import uuid
from owlready2 import *
import rdflib
from rdflib.namespace import RDF, RDFS, OWL, XSD
import types

# Path to your Protege ontology file
ontology_path = './multifactor.rdf'

# Load the ontology with rdflib
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

# Helper to get or create a class in the ontology.
# If an attribute with the given name exists and isn’t a class, a new class is created with a modified name.
def get_or_create_class(class_name):
    existing = getattr(onto, class_name, None)
    if existing is not None:
        if isinstance(existing, ThingClass):
            return existing
        else:
            # Conflict: attribute exists but isn’t a class. Create a new unique class.
            new_class_name = class_name + "_class"
            cls = types.new_class(new_class_name, (Thing,))
            setattr(onto, new_class_name, cls)
            return cls
    else:
        cls = types.new_class(class_name, (Thing,))
        setattr(onto, class_name, cls)
        return cls

# Helper to get or create a data property in the ontology
def get_or_create_data_property(prop_name, domain_cls, range_type):
    prop = getattr(onto, prop_name, None)
    if prop is None:
        prop = types.new_class(prop_name, (DataProperty,))
        prop.domain = [domain_cls]
        prop.range = [range_type]
        setattr(onto, prop_name, prop)
    return prop

with onto:
    # Process each individual and assign literal values by creating instances.
    for individual in individuals:
        parts = individual.split("#")
        if len(parts) < 2:
            continue  # Skip if the IRI is not in the expected format.
        individual_name = parts[1]
        # Get or create the class for the individual
        individual_class = get_or_create_class(individual_name)
        # Create an instance of the class
        ind_instance = individual_class()
        
        # For each triple associated with the individual, create data properties and assign values.
        for s, p, o in g.triples((individual, None, None)):
            if isinstance(o, rdflib.Literal):
                parts_prop = p.split("#")
                if len(parts_prop) < 2:
                    continue
                property_name = parts_prop[1]
                property_value = o.toPython()
                
                if isinstance(property_value, str):
                    property_type = str
                elif isinstance(property_value, bool):
                    property_type = bool
                elif isinstance(property_value, int):
                    property_type = int
                else:
                    continue  # Skip unsupported types
                
                # Get or create the data property attached to this class
                dp = get_or_create_data_property(property_name, individual_class, property_type)
                # Use .append() for non-functional properties.
                try:
                    getattr(ind_instance, property_name).append(property_value)
                except (ValueError, AttributeError):
                    setattr(ind_instance, property_name, property_value)

    # Add SWRL rules using the registered class
    new_rule = Imp()
    # Get or create the base class used in your SWRL rule
    base_cls = get_or_create_class("id_5b95bc47726e4ec58c1d58e25e3c0f45__18")
    # Get or create the attacker data property on the base class
    attacker_prop = get_or_create_data_property("attacker", base_cls, bool)
    new_rule.set_as_rule("""id_5b95bc47726e4ec58c1d58e25e3c0f45__18(?u) ^ screenLength(?u, ?l) ^ notEqual(?l, 1) -> attacker(?u, true)""")
    
    # OPTIONAL: Create an instance of the base class with a screenLength to test the rule.
    if not list(base_cls.instances()):
        test_instance = base_cls()
        # Create or get the screenLength property on base_cls.
        screen_prop = get_or_create_data_property("screenLength", base_cls, int)
        test_instance.screenLength = 2  # value > 1 triggers the rule

# Run the reasoner
sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True)

# Check the instances of base_cls for the attacker property set by the rule.
cls_18 = getattr(onto, "id_5b95bc47726e4ec58c1d58e25e3c0f45__18", None)
if cls_18:
    for instance in cls_18.instances():
        print("Instance:", instance, "attacker property:", instance.attacker)
        if instance.attacker == True:
            print("ATTACKER FOUND for instance", instance)