import rdflib
from rdflib.namespace import XSD

def mark_subject_as_attacker(graph, subj, EX):
    graph.set((subj, EX.attacker, rdflib.Literal(True, datatype=XSD.boolean)))

def mark_suspicious(graph, subj, EX):
    current_values = list(graph.objects(subj, EX.suspicious))
    current_value = current_values[0].toPython()
    graph.set((subj, EX.suspicious, rdflib.Literal(current_value+1, datatype=XSD.integer)))

def apply_rules(rdf_path: str = "./multifactor.rdf", rules=None):
    graph = rdflib.Graph()
    graph.parse(rdf_path, format="xml")
    EX = rdflib.Namespace("http://example.org/multifactor#")
    processed = set()
    for subj in graph.subjects():
        if subj in processed:
            continue
        processed.add(subj)
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
            "property": "Battery",
            "condition": lambda val: val < 95 and val > 90,
            "action": mark_suspicious
        },
        {
            "property": "latitude",
            "condition": lambda val: val < 70 and val > 60,
            "action": mark_suspicious
        },
        {
            "property": "longitude",
            "condition": lambda val: val < 130 and val > 128,
            "action": mark_suspicious
        },
        {
            "property": "availableMemory",
            "condition": lambda val: val > 952155512 and val < 1135959614,
            "action": mark_suspicious
        },
        {
            "property": "rssi",
            "condition": lambda val: val < -50 and val > -20,
            "action": mark_suspicious
        },
        {
            "property": "Processors",
            "condition": lambda val: val < 5 and val > 3,
            "action": mark_suspicious
        },
        {
            "property": "accel",
            "condition": lambda val:  val != 1,
            "action": mark_suspicious
        },
        {
            "property": "gyro",
            "condition": lambda val: val < 5 and val > 3,
            "action": mark_suspicious
        },
        {
            "property": "magnet",
            "condition": lambda val: val < 3 and val > 1,
            "action": mark_suspicious
        },
        {
            "property": "screenWidth",
            "condition": lambda val: val != 1440,
            "action": mark_suspicious
        },
        {
            "property": "screenLength",
            "condition": lambda val: val != 2872,
            "action": mark_suspicious
        },
        {
            "property": "screenDensity",
            "condition": lambda val: val != 560,
            "action": mark_suspicious
        },
        {
            "property": "hasTouchScreen",
            "condition": lambda val: val == False,
            "action": mark_suspicious
        },
        {
            "property": "hasCamera",
            "condition": lambda val: val == False,
            "action": mark_suspicious
        },
        {
            "property": "hasFrontCamera",
            "condition": lambda val: val == False,
            "action": mark_suspicious
        },
        {
            "property": "hasMicrophone",
            "condition": lambda val: val == False,
            "action": mark_suspicious
        },
        {
            "property": "hasTemperatureSensor",
            "condition": lambda val: val == False,
            "action": mark_suspicious
        },
        {
            "property": "suspicious",
            "condition": lambda val: val > 7,
            "action": mark_subject_as_attacker
        }
    ]
    apply_rules("./multifactor.rdf", rules)
