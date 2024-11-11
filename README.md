# System-logs-anomaly-detector

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