import os

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
    "looking_or_moving_around (e.g. searching)": 19,
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

def read_split_file(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def process_dataset(dataset_file, split_classes, output_file, is_test=False):
    split_indices = [activity_mapping[cls] for cls in split_classes]
    
    with open(dataset_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            parts = line.strip().split()
            if len(parts) >= 2:
                activity = int(parts[-1])  # The activity is the last item
                if activity in split_indices:
                    if is_test:
                        # Remap activities from 0 to 9 for test sets
                        remap = {old: new for new, old in enumerate(split_indices)}
                        activity = remap[activity]
                    # Replace the last item (activity) with the new activity
                    new_line = ' '.join(parts[:-1] + [str(activity)])
                    f_out.write(f"{new_line}\n")

def create_zero_shot_splits(dataset_file, splits_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    os.makedirs(output_dir, exist_ok=True)

    for i in range(10):
        seen_file = os.path.join(splits_dir, f'midlevel_seen_classes_{i}.txt')
        unseen_test_file = os.path.join(splits_dir, f'midlevel_unseen_classes_test_{i}.txt')
        unseen_val_file = os.path.join(splits_dir, f'midlevel_unseen_classes_val_{i}.txt')
        
        seen_classes = read_split_file(seen_file)
        unseen_test_classes = read_split_file(unseen_test_file)
        unseen_val_classes = read_split_file(unseen_val_file)

        train_output = os.path.join(output_dir, f'data_zero_shot_train_{i}.txt')
        val_output = os.path.join(output_dir, f'data_zero_shot_val_{i}.txt')
        test_output = os.path.join(output_dir, f'data_zero_shot_test_{i}.txt')

        process_dataset(dataset_file, seen_classes, train_output,is_test=False)
        process_dataset(dataset_file, unseen_test_classes, test_output,is_test=True)
        process_dataset(dataset_file, unseen_val_classes, val_output,is_test=False)

        print(f"Created split {i}:")
        print(f"  Train: {train_output}")
        print(f"  Val: {val_output}")
        print(f"  Test: {test_output}")
        print(f"  Seen Classes: {', '.join(seen_classes)}")
        print(f"  Unseen Classes: {', '.join(unseen_test_classes)}")
        print()


if __name__ == "__main__":
    dataset_file = "/home/bas06400/daa/all_clips.txt"
    splits_dir = "/home/bas06400/zs-drive_and_act/splits"
    output_dir = "/home/bas06400/daa/zero_shot_splits"

    create_zero_shot_splits(dataset_file, splits_dir, output_dir)