import openai
import base64
import time
import json

# === CONFIGURATION ===
IMAGE_PATH = "images/tidybot_sim_8objs.png"
API_KEY = "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"  # Replace with your key
MODEL_NAME = "chatgpt-4o-latest"
TEMPERATURE = 0.1
TOP_P = 0.7
MAX_TOKENS = 4096 * 3

# === INITIALIZE CLIENT ===
client = openai.OpenAI(api_key=API_KEY)

# === LOAD IMAGE ===
with open(IMAGE_PATH, "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

# === HELPER FUNCTION ===
def send_prompt(full_prompt, step_id):
    print(f"\n===== STEP {step_id} =====")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": full_prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{image_base64}"
                    }},
                ],
            }
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
    )
    result = response.choices[0].message.content
    print(result)
    return result

# === CONVERSATION HISTORY ===
history = []

def append_and_send(new_query, step_id):
    # Combine full conversation history so far
    full_prompt = ""
    for i, (q, r) in enumerate(history):
        full_prompt += f"\n### Step {i+1}: {q.strip()}\n\nResponse:\n{r.strip()}\n"
    full_prompt += f"\n### Step {step_id}: {new_query.strip()}\n"
    
    response = send_prompt(full_prompt, step_id)
    history.append((new_query, response))
    time.sleep(1)
    return response

# === STEP 1: Extract scene constraints ===
query_1 = """
Given the image, identify the coordinate system origin and the 3D bounding box of the shelf. 
The shelf has the following dimensions:

Depth (x): 0.4 m

Width (y): 0.2 m

Height (z): 0.4 m

The object to place is a cup with dimensions (0.08, 0.08, 0.2) meters. 
The 0.2 m side aligns with the +z axis (upright orientation).

Constraints: 
- No cup-cup collision (add a small safety gap if needed)
- No robot-cup collision (e.g., the robot trajectory should avoid colliding with any existing cups in the cupboard)
- No cupboard-cup collision (e.g., the cup should not collide with any sides of the cupboard)

Please confirm:
- Origin of the coordinate system (position and orientation)
- Direction of +x, +y, and +z
- Shelf position and size in the robot's coordinate frame
- The coordinates of the 8 corners of the cupboard 
- where should I start placing cups to avoid collisions with the robot?

"""
response_1 = append_and_send(query_1, step_id=1)

# === STEP 2: Confirm cup dimensions ===
query_2 = """

Task:
Please compute the 3D placement coordinates for 8 cups placed sequentially inside the shelf (from cup 1 to cup 8), ensuring:
- No cup-cup collision (add a small safety gap if needed)
- No robot-cup collision (e.g., the robot trajectory should avoid colliding with any existing cups in the cupboard)
- No cupboard-cup collision (e.g., the cup should not collide with any sides of the cupboard)


The x axis represents the depth and the y axis represents the width. \n 

Please assume the cups are placed on the bottom shelf. \n 

Please first calculate how many rows and colums are needed and then calculate the placement parameters. \n

For the output, could you return me a list of placement parameters for all cupds in JSON format. \n

"""
response_2 = append_and_send(query_2, step_id=2)

# === STEP 3: Safety check ===
query_3 = f"""
Please validate the placement result from Step 2.

Check for every:
- No cup-cup collisions 
- No robot-cup collision 
- No cupboard-cup collision 

If any placements are unsafe, return corrected placements. Otherwise, confirm all is valid.
"""
response_3 = append_and_send(query_3, step_id=3)

# === STEP 4: Ensure structured JSON output ===
query_4 = f"""
Please return the final validated cup placements from Step 3 as a structured list of JSON objects, 
where each object corresponds to one cup with the following schema:

{{
    "cup_id": int,               // Cup number from 1 to 8
    "position": {{
        "x": float,              // Depth (meters)
        "y": float,              // Width (meters)
        "z": float               // Height (meters)
    }}
}}

Ensure the output is a pure JSON list (no markdown, no extra text, no code formatting), like:

[
    {{
        "cup_id": 1,
        "position": {{"x": 0.35, "y": 0.05, "z": 0.0}}
    }},
    ...
]

Only return the JSON list.
"""
response_4 = append_and_send(query_4, step_id=4)

# === SAVE JSON OUTPUT ===
try:
    # Extract JSON from the response (remove any markdown formatting)
    json_text = response_4.strip()
    if json_text.startswith('```json'):
        json_text = json_text[7:]
    if json_text.endswith('```'):
        json_text = json_text[:-3]
    json_text = json_text.strip()
    
    # Parse and validate JSON
    vlm_data = json.loads(json_text)
    
    # Save to file
    output_file = "vlm_target_locations.json"
    with open(output_file, 'w') as f:
        json.dump(vlm_data, f, indent=2)
    
    print(f"\n===== VLM OUTPUT SAVED =====")
    print(f"Target locations saved to: {output_file}")
    print(f"Number of cups: {len(vlm_data)}")
    print("JSON content:")
    print(json.dumps(vlm_data, indent=2))
    
except json.JSONDecodeError as e:
    print(f"Error parsing JSON from VLM response: {e}")
    print("Raw response:")
    print(response_4)
except Exception as e:
    print(f"Error saving VLM output: {e}")