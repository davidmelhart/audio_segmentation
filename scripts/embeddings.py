import os
import torch
import cv2
from transformers import VideoMAEImageProcessor, VideoMAEModel
import numpy as np
import json

def append_embedding_to_jsonl(file_path, video_name, embedding):
    record = {
        "video": video_name,
        "embedding": embedding.tolist()  # convert NumPy to regular list
    }
    with open(file_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def resize_and_center_crop(frame, target_size=(224, 224)):
    target_width, target_height = target_size

    # Resize so width = 224, maintain aspect ratio
    original_height, original_width = frame.shape[:2]
    scale = target_width / original_width
    new_height = int(original_height * scale)

    resized = cv2.resize(frame, (target_width, new_height))

    # Center crop vertically to 224 height
    top = max(0, (new_height - target_height) // 2)
    cropped = resized[top:top + target_height, :, :]

    return cropped

def load_video_frames(video_path, num_frames=16, target_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < num_frames:
        raise ValueError(f"Video too short: requires at least {num_frames} frames")

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if idx in frame_indices:
            frame = resize_and_center_crop(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    return frames

def extract_embedding(video_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    model_id = "MCG-NJU/videomae-base"
    feature_extractor = VideoMAEImageProcessor.from_pretrained(model_id)
    model = VideoMAEModel.from_pretrained(model_id).to(device)

    frames = load_video_frames(video_path, num_frames=16)
    inputs = feature_extractor(frames, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()  # [batch_size, seq_len, hidden] → [hidden]

    return embedding.cpu().numpy()

if __name__ == "__main__":
    for filename in os.listdir(os.path.join('data', 'segments')):
        print(filename)
        video_path = os.path.join('data', 'segments', filename)
        embedding = extract_embedding(video_path)
        append_embedding_to_jsonl('output/embeddings.jsonl', filename, embedding)
