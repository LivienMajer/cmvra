import os
import re
pattern = 'test'
files = {
    'rgb': f'/home/bas06400/daa/kinect_color_{pattern}.txt',  
    'ir': f'/home/bas06400/daa/kinect_ir_{pattern}.txt',      
    'modality3': f'/home/bas06400/daa/kinect_depth_mp4_{pattern}.txt',  
    'modality4': f'/home/bas06400/daa/openpose_3d_{pattern}.txt',
    'ceiling': f'/home/bas06400/daa/ceiling_{pattern}.txt',
    'inner_mirror': f'/home/bas06400/daa/inner_mirror_{pattern}.txt',
    'a_column_co_driver': f'/home/bas06400/daa/a_column_co_driver_{pattern}.txt',
    'a_column_driver': f'/home/bas06400/daa/a_column_driver_{pattern}.txt',
    'steering_wheel': f'/home/bas06400/daa/steering_wheel_{pattern}.txt'       
}

# Common root to remove from paths.
common_root = '/home/bas06400/daa/'

# Initialize activity mapping, potentially from an existing mapping.
# Replace {} with your existing mapping if available, e.g., {'activity1': 0, 'activity2': 1}
activity_mapping = {
    "closing_door_outside": 0,
    "opening_door_outside": 1,
    "entering_car": 2,
    "closing_door_inside": 3,
    "fastening_seat_belt": 4,
    "using_multimedia_display": 5,
    "sitting_still": 6,
    "pressing_automation_button": 7,
    "fetching_an_object": 8,
    "opening_laptop": 9,
    "working_on_laptop": 10,
    "interacting_with_phone": 11,
    "drinking": 12,
    "closing_laptop": 13,
    "placing_an_object": 14,
    "unfastening_seat_belt": 15,
    "putting_on_jacket": 16,
    "opening_bottle": 17,
    "closing_bottle": 18,
    "looking_or_moving_around": 19,
    "preparing_food": 20,
    "eating": 21,
    "taking_off_sunglasses": 22,
    "putting_on_sunglasses": 23,
    "reading_newspaper": 24,
    "writing": 25,
    "talking_on_phone": 26,
    "reading_magazine": 27,
    "taking_off_jacket": 28,
    "opening_door_inside": 29,
    "exiting_car": 30,
    "opening_backpack": 31,
    "putting_laptop_into_backpack": 32,
    "taking_laptop_from_backpack": 33
}


def simplify_path(path, common_root):
    return path.replace(common_root, '')

def extract_details(path, is_skeleton=False):
    filename = os.path.basename(path)

    # Patterns for identifier extraction
    if is_skeleton:
        identifier_pattern = r"(\d+)_vp(\d+)_run(\w+)_(\d+-\d+-\d+-\d+-\d+-\d+).*?(\d+)_(\d+)\.npy"
    else:
        identifier_pattern = r"(\d+)_vp(\d+)_run(\w+)_(\d+-\d+-\d+-\d+-\d+-\d+).*?(\d+)_(\d+)\.\w+$"


    # Patterns for activity extraction
    special_case_pattern = r"([^/]+)(?=_\d+_vp\d+)"
    general_pattern = r"([a-zA-Z_]+)_\d+"

    # Attempt to match identifier
    identifier_match = re.search(identifier_pattern, filename)
    identifier = "_".join(identifier_match.groups()) if identifier_match else None

    # Attempt to match activity if not a skeleton file
    activity = None
    if not is_skeleton:
        # Attempt to match special case activity
        special_match = re.search(special_case_pattern, filename)
        activity = special_match.group(1) if special_match else None

        # If no special case, attempt general pattern
        if not activity:
            general_match = re.match(general_pattern, filename)
            activity = general_match.group(1) if general_match else None

        # Clean up activity if it's found in the special case
        if activity:
            activity = re.sub(r"\s*\([^)]*\)\s*$", "", activity).strip()

    # Report issues if matches aren't found
    if not identifier_match:
        print(f"Identifier failed to match: {filename}")
    if not activity and not is_skeleton:
        print(f"Activity failed to match: {filename}")
    #print(identifier)

    return identifier, activity


def add_to_mapping(activity):
    if activity not in activity_mapping:
        activity_mapping[activity] = len(activity_mapping)

def process_files(file_paths, common_root):
    modality_identifiers = {modality: {} for modality in file_paths}
    total_files = 0
    read_files = 0
    unmatched_identifiers = set()
    duplicate_identifiers = set()

    for modality, file_path in file_paths.items():
        seen_identifiers = set()  # Track identifiers for this modality to check for duplicates
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    total_files += 1
                    simplified_path = simplify_path(line.strip(), common_root)
                    identifier, activity = extract_details(simplified_path, is_skeleton=(modality == 'modality4'))
                    if identifier:
                        if identifier in seen_identifiers:
                            duplicate_identifiers.add(identifier)
                            print(f"Duplicate identifier found: {identifier} in {modality}")
                        seen_identifiers.add(identifier)

                        if (activity or modality == 'modality4') and identifier not in modality_identifiers[modality]:
                            modality_identifiers[modality][identifier] = simplified_path
                            if activity:
                                add_to_mapping(activity)
                            read_files += 1
                        else:
                            unmatched_identifiers.add(identifier)
                    else:
                        print(f"Failed to process file: {simplified_path}")
        except Exception as e:
            print(f"Failed to read file {file_path}: {e}")

    print(f"Total files: {total_files}, Successfully read: {read_files}")
    print(f"Unmatched Identifiers: {len(unmatched_identifiers)}")
    print(f"Duplicate Identifiers: {len(duplicate_identifiers)}")
    
    # Check for missing matches across modalities
    all_identifiers = set()
    for ids in modality_identifiers.values():
        all_identifiers.update(ids.keys())

    missing_matches = {modality: [] for modality in file_paths}
    for identifier in all_identifiers:
        for modality, ids in modality_identifiers.items():
            if identifier not in ids:
                missing_matches[modality].append(identifier)

    for modality, missing in missing_matches.items():
        print(f"Missing matches in {modality}: {len(missing)}")

    return modality_identifiers

modality_identifiers = process_files(files, common_root)

combined_dataset = []
for identifier, rgb_path in modality_identifiers['rgb'].items():
    combined_line = [rgb_path]
    for modality in files:
        if modality != 'rgb' and identifier in modality_identifiers[modality]:
            combined_line.append(modality_identifiers[modality][identifier])
    activity = extract_details(rgb_path)[1]  # Assuming activity can be extracted from the RGB modality.
    activity_id = activity_mapping.get(activity, 0)
    combined_dataset.append(" ".join(combined_line) + f" {activity_id}")

with open(f'daa_split_{pattern}_full.txt', 'w') as out_file:
    for line in combined_dataset:
        out_file.write(line + "\n")

with open('activity_mapping.txt', 'w') as map_file:
    for activity, activity_id in activity_mapping.items():
        map_file.write(f"{activity}: {activity_id}\n")

print("Dataset created with labels and activity mapping saved.")