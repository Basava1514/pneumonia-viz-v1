import os

def prepare_binary_labels(data_root):
    """
    Prepares a dictionary mapping image filenames to binary labels.
    0 = NORMAL, 1 = PNEUMONIA
    """
    categories = ["NORMAL", "PNEUMONIA"]
    label_dict = {}

    for label_name in categories:
        folder_path = os.path.join(data_root, label_name)
        if not os.path.exists(folder_path):
            continue

        for fname in os.listdir(folder_path):
            if fname.lower().endswith(('.jpeg', '.jpg', '.png')):
                label_dict[fname] = 1 if label_name.upper() == "PNEUMONIA" else 0

    return label_dict
