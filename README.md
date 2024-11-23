# System Logs Anomaly Detector - CodZilla

## Project Description 🚀

This project was developed during the 12th edition of the EESTEC OLYMPICS Hackathon, in November 2024. Our team's goal was to classify Windows system logs to identify abnormal behaviors using neural networks. We extracted JSON events from pcap files, which were then used to train and test our model.

### Team Composition and Roles 🧑‍💻👩‍💻
- **Maxim**: Developed the PCAP to JSON Processor Script, transforming pcap files into analyzable JSON format.
- **Vlada and Loredana**: Worked on Data Engineering and the Neural Network Model using PyTorch, focusing on the classification of the system log events.
- **Together**: Presented our project and achieved an impressive 90% accuracy with our model, demonstrating the effectiveness of our data processing and machine learning techniques.

### Environment and Setup 📂
The project was set up and tested in an isolated Ubuntu container, structured as follows:
- `./InputData/train`: Contains training data files (labeled).
- `./InputData/test`: Contains test data files (unlabeled).
- `./source`: Contains the main project files, including `startScript.sh` which runs the project.
- `./output/labels`: The output directory where the JSON file with the predicted labels for each test file is saved.

### Constraints and Execution ⏳
The execution within the container was limited to 2 minutes during the testing part, so our project is focused on optimizing the processing of data efficiently. The package installation phase is separate and does not affect this time limit.

### Output 📄
The output of the project is a JSON file located in `./output/labels`. This file lists the predicted label for each test file, providing insights into potential security threats detected in the system logs.

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

## Data Preprocessing

## Model
### Neural Network Model Description
For this hackathon, we utilized a standard encoding machine with PyTorch to classify Windows system logs to identify abnormal behavior. We designed a neural network called SimpleNN with multiple layers to process data efficiently:

- *Input Layer*: Receives data transformed into PyTorch tensors. The size of the input layer corresponds to the number of features in the dataset.
- *Hidden Layers and Non-linear Activation*: Our model includes several fully connected (dense) layers that progressively reduce the number of neurons from 350 to 16. Each dense layer is interspersed with a ReLU activation function to introduce non-linearity and prevent the vanishing gradient problem.
- *Dropout Layer*: To combat overfitting, a dropout layer with a dropout rate of 0.5 is integrated before the final layers, effectively dropping units randomly during training to help generalize the model better.
- *Output Layer*: The final dense layer has units equal to the number of classes to classify each log as normal or abnormal behavior.

### Training and Evaluation Process
- *Data Loading*: We load data using PyTorch's DataLoader, with separate loaders for training and testing datasets. This facilitates efficient batch processing and shuffling.
- *Training Loop*: We train the model over 110 epochs. For each epoch, the model undergoes forward and backward passes where we adjust the weights using the Adam optimizer, known for its efficiency in handling sparse gradients and adaptive learning rate management.
- *Loss Function*: Cross-Entropy Loss is employed to measure the performance of the classification, which is suitable for multi-class classification problems.
- *Evaluation Metrics*: After training, we evaluate the model's effectiveness using the accuracy metric, providing insights into the model's performance across different categories of logs.

### Anomaly Detection
In addition to the neural network, we plan to employ DBSCAN clustering algorithm as a secondary measure to identify outliers or anomalous logs in the dataset. This unsupervised technique labels data points as anomalies based on their density, further enhancing the robustness of the system against unusual patterns.

This setup ensures that our model not only learns effectively from the labeled training data but is also capable of handling new, unseen test data within the constraints of a secure, isolated environment without internet access.
