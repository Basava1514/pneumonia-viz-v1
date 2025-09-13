# utils/feedback.py

import os
import json
from datasets import Dataset
from PIL import Image
import torch

def handle_feedback(feedback_dir, image_name, prediction, prob, feedback, comments):
    os.makedirs(feedback_dir, exist_ok=True)
    feedback_data = {
        "image": image_name,
        "prediction": prediction,
        "confidence": float(prob),
        "feedback": feedback,
        "comments": comments
    }
    feedback_path = os.path.join(feedback_dir, image_name + ".json")
    with open(feedback_path, "w") as f:
        json.dump(feedback_data, f, indent=4)
    return feedback_path


def retrain_from_feedback(model, processor, feedback_dir, device):
    from transformers import Trainer, TrainingArguments
    from transformers import EarlyStoppingCallback

    entries = []
    for fname in os.listdir(feedback_dir):
        if fname.endswith(".json"):
            with open(os.path.join(feedback_dir, fname)) as f:
                data = json.load(f)
                if data["feedback"] == "No":
                    label = 0 if data["prediction"] == "Pneumonia" else 1
                    entries.append({
                        "image": os.path.join(feedback_dir, data["image"]),
                        "label": label
                    })

    if len(entries) == 0:
        return "✅ No negative feedback to retrain."

    dataset = Dataset.from_list(entries)

    def preprocess(example):
        img_path = example["image"]
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB").resize((224, 224))
            inputs = processor(images=img, return_tensors="pt")
            example["pixel_values"] = inputs["pixel_values"][0]
        else:
            example["pixel_values"] = torch.zeros((3, 224, 224))
        return example

    dataset = dataset.map(preprocess)
    dataset.set_format(type="torch", columns=["pixel_values", "label"])

    args = TrainingArguments(
        output_dir="./retrained_vit",
        per_device_train_batch_size=4,
        num_train_epochs=2,
        learning_rate=1e-5,
        save_strategy="no",
        logging_steps=10,
        disable_tqdm=False
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=processor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()
    model.save_pretrained("vit_pneumonia.pt")
    return "✅ Model retrained with feedback."
