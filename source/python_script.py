import dpkt
import pandas as pd
import json
import os

def run_json(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    def process_pcap_file(filepath, output_filepath):
        records = ""

        with open(filepath, 'rb') as f:
            pcap = dpkt.pcap.Reader(f)
            for ts, buf in pcap:
                try:
                    ip = dpkt.ip.IP(buf)
                    if isinstance(ip, dpkt.ip.IP):
                        if isinstance(ip.data, dpkt.tcp.TCP):
                            tcp = ip.data
                            if tcp.data:
                                payload_data = tcp.data

                                try:
                                    payload_text = payload_data.decode('utf-8')
                                    payload_text = payload_text.replace('\n', '')
                                except UnicodeDecodeError:
                                    payload_text = None  
                                if payload_text:
                                    records += payload_text

                except Exception as e:
                    print(f"Error parsing packet as IP in {filepath}: {e}")

        try:
            data = json.loads(records)
        except json.JSONDecodeError:
            data = {}

        def flatten_json(y, prefix=''):
            flattened = {}
            for key, value in y.items():
                full_key = f"{prefix}{key}" if prefix else key
                if isinstance(value, dict):
                    flattened.update(flatten_json(value, full_key + '.'))
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        flattened.update(flatten_json({str(i): item}, full_key + '.'))
                else:
                    flattened[full_key] = value
            return flattened

        flattened_data = flatten_json(data)

        with open(output_filepath, 'w') as json_file:
            json.dump(flattened_data, json_file, indent=4)

    count = 0
    for filename in os.listdir(input_folder):
        count += 1
        if filename.endswith(''):
            filepath = os.path.join(input_folder, filename)
            output_filename = f"{os.path.splitext(filename)[0]}.json"
            output_filepath = os.path.join(output_folder, output_filename)
            process_pcap_file(filepath, output_filepath)

run_json('/usr/src/app/InputData/test', '/usr/src/app/source/test')
run_json('/usr/src/app/InputData/train', '/usr/src/app/source/train')