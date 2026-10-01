import csv
import json

# File paths
mirna_csv_path = 'link_prediction_gat_before_adding_plots/data/miRNA_embeddings_epoch111.csv'
disease_csv_path = 'link_prediction_gat_before_adding_plots/data/disease_embeddings_256_56.csv'
association_csv_path = 'link_prediction_gat_before_adding_plots/data/merged_all_data.csv'
output_json_path = 'link_prediction_gat_before_adding_plots/data/miRNA_disease_network.json'

# Function to read embeddings from a CSV file
def read_embeddings(file_path):
    embeddings = {}
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # Skip the header
        for row in reader:
            name = row[0]
            embedding = list(map(float, row[1:]))
            embeddings[name] = embedding
    return embeddings

# Read miRNA and disease embeddings
mirna_embeddings = read_embeddings(mirna_csv_path)
disease_embeddings = read_embeddings(disease_csv_path)

# Print embeddings to debug
'''print("miRNA Embeddings:", list(mirna_embeddings.items())[:5])  # Print first 5 for brevity
print("Disease Embeddings:", list(disease_embeddings.items())[:5])'''

# Read miRNA-disease associations and create the JSON structure
relationships = []
with open(association_csv_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    headers = next(reader)  # Skip the header
    for row in reader:
        mirna_name = row[0]
        disease_name = row[1]

        # Debugging prints to ensure correct data extraction
        '''print(f"Processing association: {mirna_name} - {disease_name}")'''

        mirna_embedding = mirna_embeddings.get(mirna_name, [])
        disease_embedding = disease_embeddings.get(disease_name, [])

        # Debugging prints to ensure correct embedding extraction
        '''print(f"miRNA Embedding: {mirna_embedding}")
        print(f"Disease Embedding: {disease_embedding}")'''

        relationship = {
            "miRNA": {
                "properties": {
                    "name": mirna_name,
                    "embedding": mirna_embedding
                }
            },
            "relation": {
                "type": "ASSOCIATED_WITH"
            },
            "disease": {
                "properties": {
                    "name": disease_name,
                    "embedding": disease_embedding
                }
            }
        }
        relationships.append(relationship)

# Save to JSON file
with open(output_json_path, 'w') as json_file:
    json.dump(relationships, json_file, indent=2)

print(f"JSON file saved to {output_json_path}")
