import rdflib
from rdflib.namespace import RDF, RDFS, OWL, XSD
from owlready2 import *

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
for s, p, o in g.triples((None, RDFS.subClassOf, phones_class)):
    print(s.split("#")[1])

# Load the ontology using owlready2
ontology = get_ontology(ontology_path).load()
valid_id = ontology.search_one(iri="*id_5b95bc47726e4ec58c1d58e25e3c0f45")

# Add SWRL rules to the ontology
with ontology:
    new_rule = Imp()
    attacker_class = type("attacker", (DataProperty,), {})
    attacker_class.range = [bool]
    cls_18 = getattr(ontology, "id_5b95bc47726e4ec58c1d58e25e3c0f45__18", None)
    new_rule.set_as_rule("""http://example.org/multifactor#id_5b95bc47726e4ec58c1d58e25e3c0f45(?u) ^ http://example.org/multifactor#longitude(?u, ?l) ^ notEqual(?l, 6) -> http://example.org/multifactor#attacker(?u, true)""")
sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True)

attacker = ontology.search_one(iri="http://example.org/multifactor#attacker")
print(attacker)
if attacker and attacker.isAttacker:
    print("Attacker is classified as an attacker.")
else:
    print("Attacker is not classified as an attacker.")