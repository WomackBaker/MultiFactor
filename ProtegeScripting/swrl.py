import rdflib
from rdflib.namespace import XSD

def mark_subject_as_attacker(graph, subj, EX):
    graph.set((subj, EX.attacker, rdflib.Literal(True, datatype=XSD.boolean)))

def mark_subject_as_suspicious(graph, subj, EX):
    graph.set((subj, EX.suspicious, rdflib.Literal(True, datatype=XSD.boolean)))

def apply_rules(rdf_path: str = "./multifactor.rdf", rules=None):
    graph = rdflib.Graph()
    graph.parse(rdf_path, format="xml")
    EX = rdflib.Namespace("http://example.org/multifactor#")
    for subj in graph.subjects():
        for rule in rules:
            prop = EX[rule["property"]]
            condition_func = rule["condition"]
            action_func = rule["action"]
            values = list(graph.objects(subj, prop))
            if not values:
                continue
            for val_literal in values:
                val_python = val_literal.toPython()
                if condition_func(val_python):
                    action_func(graph, subj, EX)
    graph.serialize(destination="./multifactor.rdf", format="xml")
    print("Rules applied and graph updated.")

if __name__ == "__main__":
    rules = [
        {
            "property": "longitude",
            "condition": lambda val: val != 1,
            "action": mark_subject_as_attacker
        }
    ]
    apply_rules("./multifactor.rdf", rules)
