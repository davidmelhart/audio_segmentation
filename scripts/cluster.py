import json
import numpy as np
from sklearn.cluster import KMeans

# Load embeddings and video names
video_names = []
embedding_vectors = []

with open("output/embeddings.jsonl", "r") as f:
    for line in f:
        record = json.loads(line)
        video_names.append(record["video"])
        embedding_vectors.append(np.array(record["embedding"]))

embedding_matrix = np.stack(embedding_vectors)

# Cluster with KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(embedding_matrix)

# Save cluster assignments
with open("output/clusters.jsonl", "w") as out_f:
    for video, cluster in zip(video_names, cluster_labels):
        out_f.write(json.dumps({"video": video, "cluster": int(cluster)}) + "\n")
