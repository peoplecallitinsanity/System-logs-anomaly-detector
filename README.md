# System-logs-anomaly-detector

## Project Decription
## Project Description 🚀

This project was developed during the 12th edition of the EESTEC OLYMPICS Hackathon. Our team's goal was to classify Windows system logs to identify abnormal behaviors using neural networks. We extracted JSON events from pcap files, which were then used to train and test our model.

### Team Composition and Roles 🧑‍💻👩‍💻
- **Maxim**: Developed the PCAP to JSON Processor Script, transforming pcap files into analyzable JSON format.
- **Vlada and Loredana**: Worked on Data Engineering and the Neural Network Model using PyTorch, focusing on the classification of the system log events.

### Environment and Setup 📂
The project is set up in an isolated Ubuntu container, structured as follows:
- `/usr/src/app/DataFolder`: Includes all necessary packages for offline installation via `packageScript.sh`.
- `/usr/src/app/InputData/train`: Contains training data files (labeled).
- `/usr/src/app/InputData/test`: Contains test data files (unlabeled).
- `/usr/src/app/source`: Contains the main project files, including `startScript.sh` which runs the project.
- `/usr/src/app/output/labels`: The output directory where the JSON file with the predicted labels for each test file is saved.

### Constraints and Execution ⏳
The execution within the container is limited to 2 minutes, focusing on optimizing the processing of data efficiently. The package installation phase is separate and does not affect this time limit.

### Output 📄
The output of the project is a JSON file located in `/output/labels`. This file lists the predicted label for each test file, providing insights into potential security threats detected in the system logs.

### Conclusion 🌟
This project showcases our ability to work collaboratively under pressure, developing a complex system capable of addressing real-world problems through data transformation and machine learning.



## PCAP to JSON Processor Script
This script processes PCAP files by extracting **JSON** payloads from TCP packets, cleaning up and flattening the **JSON** structure, and saving it in a specified output folder for easier handling and analysis.

### Features
* Parses **PCAP** files and extracts **JSON**-formatted data from TCP packet payloads.
* Handles nested **JSON** structures by flattening them for simplified data manipulation.
* Saves output **JSON** files with cleaned and structured data in specified output directories.
### Requirements
Ensure the following libraries are installed:
>pip install dpkt pandas

### How to Use the Script
The main function, **run_json**, processes all **.pcap** files in the specified input directories and outputs flattened **JSON** data to the output directories.

## Data Engineering 

## Model
