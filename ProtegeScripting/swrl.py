import csv
import rdflib
from rdflib.namespace import XSD
import re
#Phones and (suspicious some xsd:float[>= 8.0f])
def mark_subject_as_attacker(graph, subj, EX):
    graph.set((subj, EX.attacker, rdflib.Literal(True, datatype=XSD.boolean)))

def mark_suspicious(graph, subj, EX):
    current_values = list(graph.objects(subj, EX.suspicious))
    current_value = current_values[0].toPython() if current_values else 0
    graph.set((subj, EX.suspicious, rdflib.Literal(current_value + 1, datatype=XSD.float)))

def load_base_csv(path="base2.csv"):
    with open(path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        base_config = {}
        for row in reader:
            raw_id = row['id'].replace("-", "")
            base_config[raw_id] = row
        return base_config

def extract_baseid_from_subject(subject_uri: str):
    match = re.search(r'id_([a-fA-F0-9]+)__\d+', subject_uri)
    return match.group(1) if match else None

def build_rules_from_row(row):
    def range_rule(prop, low_key, high_key):
        low = float(row[low_key])
        high = float(row[high_key])
        return {
            "property": prop,
            "condition": lambda val, lo=low, hi=high: not (lo <= val <= hi),
            "action": mark_suspicious
        }

    def exact_match_rule(prop, key):
        val = float(row[key])
        return {
            "property": prop,
            "condition": lambda v, expected=val: v != expected,
            "action": mark_suspicious
        }

    return [
        range_rule("Battery", "Battery_low", "Battery_high"),
        range_rule("latitude", "latitude_low", "latitude_high"),
        range_rule("longitude", "longitude_low", "longitude_high"),
        range_rule("availableMemory", "availableMemory_low", "availableMemory_high"),
        range_rule("rssi", "rssi_low", "rssi_high"),
        range_rule("Processors", "Processors_low", "Processors_high"),
        range_rule("gyro", "gyro_low", "gyro_high"),
        range_rule("magnet", "magnet_low", "magnet_high"),
        exact_match_rule("screenWidth", "screenWidth"),
        exact_match_rule("screenLength", "screenLength"),
        exact_match_rule("screenDensity", "screenDensity"),
        {
            "property": "accel",
            "condition": lambda val: val != 1,
            "action": mark_suspicious
        },
        {
            "property": "hasTouchScreen",
            "condition": lambda val: val is False,
            "action": mark_suspicious
        },
        {
            "property": "hasCamera",
            "condition": lambda val: val is False,
            "action": mark_suspicious
        },
        {
            "property": "hasFrontCamera",
            "condition": lambda val: val is False,
            "action": mark_suspicious
        },
        {
            "property": "hasMicrophone",
            "condition": lambda val: val is False,
            "action": mark_suspicious
        },
        {
            "property": "hasTemperatureSensor",
            "condition": lambda val: val is False,
            "action": mark_suspicious
        },
        {
            "property": "suspicious",
            "condition": lambda val: val > 7,
            "action": mark_subject_as_attacker
        }
    ]

def apply_rules(rdf_path: str = "./multifactor.rdf", base_csv_path="base2.csv"):
    base_config = load_base_csv(base_csv_path)
    graph = rdflib.Graph()
    graph.parse(rdf_path, format="xml")
    EX = rdflib.Namespace("http://example.org/multifactor#")
    processed = set()

    for subj in graph.subjects():
        subj_str = str(subj)
        if subj_str in processed:
            continue
        if not subj_str.startswith("http://example.org/multifactor#id_"):
            continue

        processed.add(subj_str)

        baseid = extract_baseid_from_subject(subj_str)
        if not baseid:
            continue
        if baseid not in base_config:
            continue

        rules = build_rules_from_row(base_config[baseid])
        for rule in rules:
            prop = EX[rule["property"]]
            condition_func = rule["condition"]
            action_func = rule["action"]
            values = list(graph.objects(subj, prop))
            if not values:
                continue
            for val_literal in values:
                try:
                    val_python = val_literal.toPython()
                    if condition_func(val_python):
                        action_func(graph, subj, EX)
                except Exception as e:
                    print(f"Error processing {subj} {prop}: {e}")

    graph.serialize(destination=rdf_path, format="xml")
    print("Rules applied and graph updated.")

if __name__ == "__main__":
    apply_rules("./multifactor.rdf", "base2.csv")
