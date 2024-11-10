# MultiFactor Authentication Project

This repository contains various scripts and tools for context and adaptive multi-factor authentication, data processing, and machine learning. The project is organized into several directories, each serving a specific purpose.

## Directory Structure

.
├── AndroidDocker/
│   ├── Dockerfile
│   ├── multifactorapp.apk
│   ├── phone.sh
│   └── refer.py
├── csvtomysql/
│   ├── convert.py
│   └── data/
├── DataGen/
│   ├── Dockerfile
│   ├── endpoint.py
│   ├── main.py
│   ├── requirements.txt
│   └── similar.py
├── Docker/
│   ├── data/
│   ├── Dockerfile
│   ├── face.py
│   ├── feature_extraction.py
│   ├── GetData.py
│   ├── parameters.py
│   ├── preprocess.py
│   ├── README.md
│   ├── requirements.txt
│   ├── voice_auth_model_cnn/
│   ├── voice_auth.py
│   ├── voice.py
│   └── voiceface.py
├── Executable/
│   └── build/
├── Kubernetes/
├── LogDocker/
├── MultiFactorApp/
├── ProtegeScripting/
├── README.md
└── vms.txt

## Directories and Files

### AndroidDocker
Contains Docker-related files for building and running the Android multi-factor authentication app inside of a docker container.

- Dockerfile: Docker configuration file.
- multifactorapp.apk: The Android application package.
- phone.sh: Shell script for phone-related operations.

### csvtomysql
Scripts for converting CSV data to MySQL database.

- convert.py: Script to convert CSV files to MySQL.
- data/: Directory containing CSV data files.

### DataGen
Container for data generation and processing from the android container.

- Dockerfile: Docker configuration file.
- endpoint.py: Script for handling endpoints.
- main.py: Main script for data generation.
- requirements.txt: Python dependencies.
- similar.py: Script for finding similar data.

### Docker
Contains Docker-related files and scripts for various data processing and machine learning tasks.

- Dockerfile: Docker configuration file.
- face.py: Script for face recognition.
- feature_extraction.py: Script for feature extraction.
- GetData.py: Script for data retrieval.
- parameters.py: Script for handling parameters.
- preprocess.py: Script for data preprocessing.
- README.md: Documentation for the Docker directory.
- requirements.txt: Python dependencies.
- voice_auth_model_cnn/: Directory containing voice authentication model files.
- voice_auth.py: Script for voice authentication.
- voice.py: Script for voice-related operations.
- voiceface.py: Script for combined voice and face recognition.

### Executable
Contains build files for the windows executable version of the multifactor application.

- build/: Directory containing build files.

### Kubernetes
Contains Kubernetes configuration files to run the logging container, data generation container, authentication container, and the android container.

### LogDocker
Contains Docker-related files for logging from the data generation container.

### MultiFactorApp
Contains the multi-factor authentication application and mobile files.

### ProtegeScripting
Contains scripts for converting csv into Protege ontology management.

### vms.txt
Contains information about virtual machines.

## Getting Started

### Prerequisites

- Docker
- Python 3.11
- MySQL

### Installation

1. Clone the repository:
   git clone https://github.com/WomackBaker/MultiFactor.git
   cd multifactor-authentication


### Usage

1. Run the container inside of kubernetes:
   
   kubectl apply -f ./android.yaml -f auth.yaml -f gen.yaml -f log.yaml
