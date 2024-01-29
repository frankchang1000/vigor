import json
import argparse

def generate_segmentation(bbox):
    x, y, w, h = bbox
    return [[x, y, x + w, y, x + w, y + h, x, y + h]]

def fill_segmentation_values(data):
    for annotation in data['annotations']:
        bbox = annotation['bbox']
        segmentation = generate_segmentation(bbox)
        annotation['segmentation'] = segmentation

def main():
    parser = argparse.ArgumentParser(description="Fill in segmentation values in a COCO-style JSON file.")
    parser.add_argument("input_file", help="Path to the input JSON file")
    args = parser.parse_args()

    with open(args.input_file, 'r') as file:
        json_data = json.load(file)

    fill_segmentation_values(json_data)

    # Write the updated JSON data back to the file
    output_file = args.input_file.replace('.json', '_updated.json')
    with open(output_file, 'w') as file:
        json.dump(json_data, file, indent=2)

    print(f"Segmentation values filled in. Updated JSON saved to {output_file}")

if __name__ == "__main__":
    main()
