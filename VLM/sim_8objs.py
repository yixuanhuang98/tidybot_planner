import openai
import base64
import time
import json
import argparse
from datetime import datetime
from pathlib import Path

# === CONFIGURATION ===
# IMAGE_PATH = "images/overview_000000_annotated.png"
IMAGE_PATH = "images/8objs_small_2.png"
TEMPERATURE = 0.1
TOP_P = 0.7
MAX_TOKENS = 4096 * 3


def main(api_key: str, num_runs: int, model_name: str, save_runs: bool):
    """
    Main function to run the VLM pipeline for cup placement.
    """
    # === CREATE TIMESTAMPED OUTPUT DIRECTORY (optional) ===
    output_dir: Path | None = None
    if save_runs:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path("runs") / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving results to: {output_dir}")
    else:
        print("Running without saving outputs (use --save_runs to enable saving).")

    # === INITIALIZE CLIENT ===
    client = openai.OpenAI(api_key=api_key)

    # === LOAD IMAGE ===
    with open(IMAGE_PATH, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    all_runs_data = []

    for run_idx in range(num_runs):
        print(f"\n\n{'='*20} RUN {run_idx + 1}/{num_runs} {'='*20}\n")

        # === HELPER FUNCTION ===
        def send_prompt(full_prompt, step_id):
            print(f"\n===== STEP {step_id} =====")
            if "gpt-5" or "o3" in model_name:
                response = client.chat.completions.create(
                    model=model_name,
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
                    max_completion_tokens=MAX_TOKENS,
                )
            else:
                response = client.chat.completions.create(
                    model=model_name,
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

        Depth (x): 0.35 m
        Width (y): 0.3 m
        Height (z): 0.4 m

        The object to place is a cup with dimensions (0.06, 0.14, 0.2) meters. 
        The 0.2 m side aligns with the +z axis (upright orientation).
        You will output the center of these objects. 

        Please confirm:
        - Origin of the coordinate system (position and orientation)
        - Direction of +x, +y, and +z
        - Shelf position and size in the coordinate system 
        - The coordinates of the 8 corners of the cupboard 
        - where should I start placing cups to satisfy the constraints?

        Constraints: 
        - No cupboard-cup collision (e.g., the cup should not collide with any sides of the cupboard)
        - No cup-cup collision (please ensure the new cups are not colliding with any existing cups)

        """
        response_1 = append_and_send(query_1, step_id=1)

        # === STEP 2: Confirm cup dimensions ===
        query_2 = """

        Task:
        Please compute the 3D placement coordinates for 8 cups placed sequentially inside the shelf (from cup 1 to cup 8).

        Please assume the cups are placed on the bottom shelf but please ensure they are not collision with the bottom shelf. 

        Please first calculate how many rows and colums are needed and then calculate the placement parameters. 

        For the output, could you return me a list of placement parameters for all cupds in JSON format. 

        Please ensure:
        - No cup-cup collision (add a small safety gap if needed and ensure the new cups are not colliding with any existing cups)
        - No cupboard-cup collision (e.g., the cup should not collide with any sides of the cupboard)
        """
        response_2 = append_and_send(query_2, step_id=2)

        # === STEP 3: Safety check ===
        query_3 = f"""
        Please validate the placement result from Step 2.

        Check for every:
        - No cup-cup collisions 
        - No cupboard-cup collision 

        Does it follow the how you should first place the cups? 

        If any placements are unsafe, return corrected placements. Otherwise, confirm all is valid.
        """
        response_3 = append_and_send(query_3, step_id=3)

        # === STEP 4: Ensure structured JSON output ===
        query_4 = f"""
        Please return the final validated cup placements from Step 3 as a structured list of JSON objects, 
        where each object corresponds to one cup with the following schema:

        {{
            "cup_id": 1,
            "position": {{
                "x": float,
                "y": float,
                "z": float
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
            all_runs_data.append(vlm_data)
            
            # Save to file (optional)
            if save_runs and output_dir is not None:
                output_file = output_dir / f"vlm_target_locations_run_{run_idx + 1}.json"
                with open(output_file, 'w') as f:
                    json.dump(vlm_data, f, indent=2)
                print(f"\n===== VLM OUTPUT FOR RUN {run_idx + 1} SAVED =====")
                print(f"Target locations saved to: {output_file}")
            else:
                print("\n===== VLM OUTPUT (NOT SAVED) =====")

            print(f"Number of cups: {len(vlm_data)}")
            print("JSON content:")
            print(json.dumps(vlm_data, indent=2))
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from VLM response on run {run_idx + 1}: {e}")
            print("Raw response:")
            print(response_4)
        except Exception as e:
            print(f"Error handling VLM output on run {run_idx + 1}: {e}")

    if all_runs_data and save_runs and output_dir is not None:
        aggregated_output_file = output_dir / "vlm_target_locations_all_runs.json"
        with open(aggregated_output_file, 'w') as f:
            json.dump(all_runs_data, f, indent=2)
        print(f"\n\n{'='*20} ALL RUNS COMPLETED {'='*20}")
        print(f"Aggregated results saved to: {aggregated_output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VLM-based cup placement task.")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key.")
    parser.add_argument("-n", "--num_runs", type=int, default=1, help="Number of times to run the VLM pipeline.")
    parser.add_argument("--model_name", type=str, default="chatgpt-4o-latest", help="Name of the model to use.")
    parser.add_argument("--save_runs", action="store_true", help="If set, save outputs under runs/<timestamp>/.")
    args = parser.parse_args()
    main(args.api_key, args.num_runs, args.model_name, args.save_runs)