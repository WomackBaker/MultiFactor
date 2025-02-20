#!/usr/bin/env python3

import sys
import uuid
from owlready2 import *
import rdflib
from rdflib.namespace import RDF, RDFS, OWL, XSD

# Path to your Protege ontology file
ontology_path = './multifactor.rdf'
# Load the ontology using owlready2
onto = get_ontology(ontology_path).load()

onto._set_default_namespace("http://example.org/multifactor#")
onto.load()
# Create classes and data properties dynamically
with onto:
    new_rule = Imp()
    print("Classes:", list(onto.classes()))
    attacker_class = type("attacker", (DataProperty,), {})
    attacker_class.range = [bool]
    new_rule.set_as_rule(f"""Phones(?u) ^ screenLength(?u, ?l) ^ lessThan(?l, 0) -> attacker(?u, true)""")
# Run the reasoner
sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True)

test_indv = onto["id_5b95bc47726e4ec58c1d58e25e3c0f45__18"]
if hasattr(test_indv, "attacker"):
    print(f"{test_indv}.attacker: {test_indv.attacker}")