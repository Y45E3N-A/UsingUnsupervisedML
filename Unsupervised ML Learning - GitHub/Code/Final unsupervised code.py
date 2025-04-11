import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from kymatio import Scattering2D
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import umap

# Set GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU memory growth enabled.")
    except RuntimeError as e:
        print(f"⚠ Could not enable memory growth: {e}")

# Load images
def load_images_from_directory(directory, target_size=(128, 128)):
    images, filenames, filepaths = [], [], []
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(valid_extensions):
                img_path = os.path.join(root, filename)
                try:
                    img = image.load_img(img_path, target_size=target_size, color_mode='grayscale')
                    img_array = image.img_to_array(img) / 255.0
                    images.append(img_array)
                    filenames.append(filename)
                    filepaths.append(img_path)
                except Exception as e:
                    print(f"⚠ Failed to load {img_path}: {e}")
    return np.array(images), filenames, filepaths

image_dir = r"C:\Users\yasee\OneDrive\Desktop\BEng Project - Yaseen\image dataset\Datasets\New Image Dataset\Unorganised for unsupervised wo aug"
images, filenames, filepaths = load_images_from_directory(image_dir)
images = np.expand_dims(images, axis=-1)

# Apply Wavelet Scattering
scattering = Scattering2D(J=2, shape=(128, 128))

def apply_wavelet_scattering(images):
    scattering_results = np.array([scattering(img.squeeze()) for img in images])
    print(scattering_results.shape)  # Add this line to see actual dimensions
    return np.transpose(scattering_results, (0, 2, 3, 1))


images_wst = apply_wavelet_scattering(images)

# Define CNN model for feature extraction
def build_feature_extractor(input_shape=(32, 32, 81)):
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu')
    ])
    return model

# Extract features
model = build_feature_extractor()
features = model.predict(images_wst, batch_size=4)

# Standardize features
features = StandardScaler().fit_transform(features)

# Reduce dimensions with UMAP
umap_reducer = umap.UMAP(n_components=5, init='spectral', metric='cosine', random_state=42)
reduced_features = umap_reducer.fit_transform(features)

# Find optimal clusters
def find_optimal_clusters(data, max_clusters=10):
    silhouette_scores = []
    cluster_range = range(2, min(max_clusters, len(data) - 1))
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        if len(set(cluster_labels)) > 1:
            silhouette_scores.append(silhouette_score(data, cluster_labels))
    return cluster_range[np.argmax(silhouette_scores)]

optimal_clusters = find_optimal_clusters(reduced_features)
kmeans = KMeans(n_clusters=optimal_clusters, n_init=20, random_state=42)
clusters = kmeans.fit_predict(reduced_features)

# Evaluate clustering
sil_score = silhouette_score(reduced_features, clusters)
ch_score = calinski_harabasz_score(reduced_features, clusters)
db_score = davies_bouldin_score(reduced_features, clusters)

# Save clusters
output_dir = "Clustered_Images"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "cluster_assignments.txt"), "w", encoding="utf-8") as f:
    f.write("📊 Best Clustering Statistics:\n")
    f.write(f"✔ Best Silhouette Score: {sil_score:.4f}\n")
    f.write(f"✔ Best Calinski-Harabasz Index: {ch_score:.2f}\n")
    f.write(f"✔ Best Davies-Bouldin Index: {db_score:.4f}\n\n")

    for cluster_id in range(optimal_clusters):
        f.write(f"Cluster {cluster_id}:\n")
        cluster_folder = os.path.join(output_dir, f"Cluster_{cluster_id}")
        os.makedirs(cluster_folder, exist_ok=True)
        for path, label in zip(filepaths, clusters):
            if label == cluster_id:
                shutil.copy(path, cluster_folder)
                f.write(f"{os.path.basename(path)}\n")
        f.write("\n")
# Re-run UMAP to reduce from 5D to 2D for visualization
umap_vis_reducer = umap.UMAP(n_components=2, init='spectral', metric='cosine', random_state=42)
visual_features = umap_vis_reducer.fit_transform(reduced_features)

# Visualize clusters
plt.figure(figsize=(10, 7))
sns.scatterplot(x=visual_features[:, 0], y=visual_features[:, 1], hue=clusters, palette="viridis", alpha=0.8)
plt.title("Best Cluster Visualization (2D UMAP from 5D)")
plt.xlabel("UMAP Component 1")
plt.ylabel("UMAP Component 2")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "best_cluster_plot.png"))



print("✅ Clustering complete and results saved.")
