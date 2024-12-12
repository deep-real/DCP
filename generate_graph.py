import os
from openai import OpenAI
from descriptor_strings import stringtolist
import json
import itertools
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(description='Generate Knowledge Graph.')
parser.add_argument('--openai_api_key', default="", type=str, help='Openai API Key')
parser.add_argument('--descriptor_path', default="", type=str, help='Path to Descriptor')
parser.add_argument('--save_path', default="", type=str, help='Save Path')
args = parser.parse_args()
print(args)

os.environ["OPENAI_API_KEY"] = args.openai_api_key
client = OpenAI()

def generate_prompt(category_name: str):
    json_part = """
    graph = {
      "nodes": [
        {"id": "American Robin", "label": "American Robin"},
        {"id": "Breast", "label": "Breast"},
        {"id": "Tail", "label": "Tail"},
        {"id": "Beak", "label": "Beak"},
        {"id": "Eyes", "label": "Eyes"},
        {"id": "Legs", "label": "Legs"},
        {"id": "Red", "label": "Red"},
        {"id": "Gray", "label": "Gray"},
        {"id": "Yellow", "label": "Yellow"},
        {"id": "Round", "label": "Round"},
        {"id": "Thin", "label": "Thin"}
      ],
      "edges": [
        {"source": "American Robin", "target": "Breast", "relation": "has"},
        {"source": "American Robin", "target": "Tail", "relation": "has"},
        {"source": "American Robin", "target": "Beak", "relation": "has"},
        {"source": "American Robin", "target": "Eyes", "relation": "has"},
        {"source": "American Robin", "target": "Legs", "relation": "has"},
        {"source": "Breast", "target": "Red", "relation": "is"},
        {"source": "Tail", "target": "Gray", "relation": "is"},
        {"source": "Beak", "target": "Yellow", "relation": "is"},
        {"source": "Eyes", "target": "Round", "relation": "are"},
        {"source": "Legs", "target": "Thin", "relation": "are"}
      ]
    }

    graph = {
      "nodes": [
        {"id": "Airliner", "label": "Airliner"},
        {"id": "Wings", "label": "Wings"},
        {"id": "Tail", "label": "Tail"},
        {"id": "Fuselage", "label": "Fuselage"},
        {"id": "Engines", "label": "Engines"},
        {"id": "Windows", "label": "Windows"},
        {"id": "Swept-back", "label": "Swept-back"},
        {"id": "Vertical", "label": "Vertical"},
        {"id": "Long", "label": "Long"},
        {"id": "Multiple", "label": "Multiple"},
        {"id": "Rectangular", "label": "Rectangular"}
      ],
      "edges": [
        {"source": "Airliner", "target": "Wings", "relation": "has"},
        {"source": "Airliner", "target": "Tail", "relation": "has"},
        {"source": "Airliner", "target": "Fuselage", "relation": "has"},
        {"source": "Airliner", "target": "Engines", "relation": "has"},
        {"source": "Airliner", "target": "Windows", "relation": "has"},
        {"source": "Wings", "target": "Swept-back", "relation": "are"},
        {"source": "Tail", "target": "Vertical", "relation": "is"},
        {"source": "Fuselage", "target": "Long", "relation": "is"},
        {"source": "Engines", "target": "Multiple", "relation": "are"},
        {"source": "Windows", "target": "Rectangular", "relation": "are"}
      ]
    }
    """
    prompt = f"What are useful visual concepts for distinguishing a {category_name} in a photo? These features should be visually distinctable and have limited overlap with each other. These features should include attributes and their relations. For each item, you should be concise and precise, and use no more than five words. No ambiguous answers. Show your answer using a graph structure in JSON format strictly following the examples shown above. Only contains two depths of nodes (depth 1: attributes, depth 2: subattributes). The knowledge graph should only contain 5 attributes and each attribute only has one sub attribute. No other explanations, only provide the graph."
    return json_part + prompt

with open(args.descriptor_path, 'r') as file:
    data = json.load(file)
class_list = list(data.keys())

print("Number of Classes: " + str(len(class_list)))

print("Generating Prompts...")
prompts = [generate_prompt(category.replace('_', ' ')) for category in class_list]

print("Querying Openai...")
responses = [client.chat.completions.create(
  model="gpt-4-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": f"{prompt}"}
  ]
) for prompt in tqdm(prompts)]

print("Processing JSON Data...")
response_texts = [responses[i].choices[0].message.content for i in range(len(responses))]
json_data = [response_texts[i].strip().strip('`').strip('json\n').strip('/ graph').strip() for i in range(len(response_texts))]
json_objects = []
for i in tqdm(range(len(json_data))):
    requery_needed = True
    data = json_data[i]
    
    while requery_needed:
        try:
            # Attempt to parse the JSON string
            parsed_data = json.loads(data)
        except json.JSONDecodeError as e:
            # This block will run if a JSONDecodeError occurs
            print(f"Error parsing JSON: {e}")
            print("Try a new response")

            prompt = generate_prompt(class_list[i])
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"{prompt}"}
                ]
                )
            response_text = response.choices[0].message.content
            data = response_text.strip().strip('`').strip('json\n').strip('/ graph').strip()
        else:
            json_objects.append(parsed_data)
            requery_needed = False

keyed_json_objects = {cat: descr for cat, descr in zip(class_list, json_objects)}

print("Saving...")
with open(args.save_path, 'w') as file:
    json.dump(keyed_json_objects, file, indent=2)
print("File Saved. Done.")