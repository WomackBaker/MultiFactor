from owlready2 import *
import subprocess

def analyze_user(attributes):
    # Load the ontology
    onto = get_ontology(r".\multifactor.rdf").load()
    
    # Check if ontology is loaded correctly
    if not onto:
        print("Failed to load ontology.")
        return
    
    # Check if User class exists in the ontology
    if not hasattr(onto, 'Phones'):
        print("User class not found in the ontology.")
        return
    
    # Run the reasoner
    with onto:
        try:
            sync_reasoner()
        except subprocess.CalledProcessError as e:
            print(f"Error: {e.output.decode()}")
            raise
    
    # Search for users matching the given attributes
    for user in onto.Phones.instances():
        match = True
        for attr, value in attributes.items():
            if not hasattr(user, attr) or getattr(user, attr) != value:
                match = False
                break
        if match:
            return user.likelihood
    
    print("No matching user found.")
    return None


if __name__ == "__main__":
    attributes = {
        "latitude": 64.8779583,
        "longitude": 129.91628500000002,
        "ipString": "10.0.2.230",
        "currentTime": "2024-11-07 19:17:23",
        "availableMemory": "995756362",
        "rssi": "-38",
        "timezone": "Etc/UTC",
        "Processors": "4",
        "Battery": "100",
        "Vendor": "Google",
        "Model": "sdk_gphone_x86_64",
        "cpu": "ranchu",
        "accel": "1",
        "gyro": "4",
        "magnet": "2",
        "screenWidth": "1080",
        "screenLength": "2148",
        "screenDensity": "440",
        "hasTouchScreen": "True",
        "hasCamera": "True",
        "hasFrontCamera": "True",
        "hasMicrophone": "True",
        "hasTemperatureSensor": "True",
        "propertyof": "64c3330a-657b-4d19-9eb1-c0fd4ecb296d"
    }
    likelihood = analyze_user(attributes)
    if likelihood:
        print(f"Likelihood of being the user: {likelihood}%")
    else:
        print("No matching user found.")