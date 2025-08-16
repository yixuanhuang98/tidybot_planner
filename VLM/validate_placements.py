"""
Validates VLM-generated cup placement plans against environment and object constraints.

This script checks for:
1.  Object-environment collisions (is the cup inside the cupboard?).
2.  Object-object collisions (do any two cups overlap?).
3.  Placement strategy (are cups placed from back to front to avoid robot collision?).

Usage:
    python validate_placements.py <path_to_json_or_directory>
"""
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

# === GROUND TRUTH GEOMETRY ===

# Dimensions of the cupboard's inner, usable volume
CUPBOARD_SIZE = {'x': 0.35, 'y': 0.3, 'z': 0.4}  # Depth, Width, Height

# Position of the center of the cupboard's bottom face in the world frame
CUPBOARD_POSITION = {'x': 1.0, 'y': 0.0, 'z': 0.2}

# Dimensions of a single cup
CUP_DIMS = {'x': 0.06, 'y': 0.14, 'z': 0.2}

# Safety margin for collision checks
SAFETY_MARGIN = 0.01

def get_cupboard_bounds() -> Dict[str, Tuple[float, float]]:
    """Calculates the world-coordinate bounding box of the cupboard's interior."""
    half_size = {k: v / 2 for k, v in CUPBOARD_SIZE.items()}
    return {
        'x': (CUPBOARD_POSITION['x'] - half_size['x'], CUPBOARD_POSITION['x'] + half_size['x']),
        'y': (CUPBOARD_POSITION['y'] - half_size['y'], CUPBOARD_POSITION['y'] + half_size['y']),
        'z': (CUPBOARD_POSITION['z'], CUPBOARD_POSITION['z'] + CUPBOARD_SIZE['z']),
    }

def get_cup_bounds(cup_center: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
    """Calculates the world-coordinate bounding box of a cup given its center."""
    half_dims = {k: v / 2 for k, v in CUP_DIMS.items()}
    return {
        'x': (cup_center['x'] - half_dims['x'], cup_center['x'] + half_dims['x']),
        'y': (cup_center['y'] - half_dims['y'], cup_center['y'] + half_dims['y']),
        'z': (cup_center['z'] - half_dims['z'], cup_center['z'] + half_dims['z']),
    }

def check_object_environment_collision(cup_center: Dict[str, float], cupboard_bounds: Dict[str, Tuple[float, float]]) -> bool:
    """Checks if a cup is fully inside the cupboard."""
    cup_bounds = get_cup_bounds(cup_center)
    for axis in ['x', 'y', 'z']:
        if cup_bounds[axis][0] < cupboard_bounds[axis][0] or cup_bounds[axis][1] > cupboard_bounds[axis][1]:
            return True  # Collision detected
    return False

def check_object_object_collision(cup1_center: Dict[str, float], cup2_center: Dict[str, float]) -> bool:
    """Checks if two cups overlap using AABB collision detection."""
    for axis in ['x', 'y', 'z']:
        dist = abs(cup1_center[axis] - cup2_center[axis])
        min_dist = (CUP_DIMS[axis] / 2) + (CUP_DIMS[axis] / 2) + SAFETY_MARGIN
        if dist < min_dist:
            return True # Collision detected
    return False

def validate_plan(plan: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """
    Validates a single placement plan.
    Returns a tuple of (is_valid, error_messages).
    """
    errors = []
    cupboard_bounds = get_cupboard_bounds()

    # 1. Check for object-environment collisions
    for cup in plan:
        pos = cup['position']
        if check_object_environment_collision(pos, cupboard_bounds):
            errors.append(f"Cup {cup['cup_id']}: Environment collision at {pos}")

    # 2. Check for object-object collisions
    for i in range(len(plan)):
        for j in range(i + 1, len(plan)):
            pos1 = plan[i]['position']
            pos2 = plan[j]['position']
            if check_object_object_collision(pos1, pos2):
                errors.append(f"Cup {plan[i]['cup_id']} and Cup {plan[j]['cup_id']}: Object-object collision")

    # 3. Check for robot collision avoidance (heuristic: back-to-front placement)
    # The VLM was instructed to place cups to avoid robot collision, which implies
    # placing objects at the back first. We check if the x-coordinates are sorted descending.
    x_coords = [cup['position']['x'] for cup in plan]
    if not all(x_coords[i] >= x_coords[i+1] for i in range(len(x_coords)-1)):
        errors.append(f"Placement strategy invalid: Cups are not placed from back to front. X-coords: {x_coords}")

    return not errors, errors

def main(path: Path):
    """Main validation function."""
    if not path.exists():
        print(f"Error: Path does not exist: {path}")
        return

    json_files = []
    if path.is_dir():
        json_files.extend(path.glob('**/*.json'))
    elif path.is_file():
        json_files.append(path)

    total_plans = 0
    successful_plans = 0

    print(f"Found {len(json_files)} JSON file(s) to validate.")

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Handle both single and aggregated JSON formats
            if isinstance(data, list) and isinstance(data[0], list): # Aggregated file
                plans = data
            else: # Single run file
                plans = [data]

            print(f"\n--- Validating {json_file.name} ({len(plans)} plan(s)) ---")
            
            for i, plan in enumerate(plans):
                total_plans += 1
                is_valid, errors = validate_plan(plan)
                if is_valid:
                    successful_plans += 1
                    print(f"  Plan {i+1}: SUCCESS")
                else:
                    print(f"  Plan {i+1}: FAILED")
                    for error in errors:
                        print(f"    - {error}")

        except json.JSONDecodeError:
            print(f"\n--- Skipping {json_file.name}: Invalid JSON format ---")
        except Exception as e:
            print(f"\n--- An error occurred while processing {json_file.name}: {e} ---")

    if total_plans > 0:
        success_rate = (successful_plans / total_plans) * 100
        print("\n" + "="*20 + " VALIDATION SUMMARY " + "="*20)
        print(f"Total Plans Checked: {total_plans}")
        print(f"Successful Plans:    {successful_plans}")
        print(f"Success Rate:        {success_rate:.2f}%")
        print("="*62)
    else:
        print("\nNo valid plans found to validate.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate VLM-generated placement plans.")
    parser.add_argument("path", type=str, help="Path to a JSON file or a directory of JSON files.")
    args = parser.parse_args()
    main(Path(args.path)) 