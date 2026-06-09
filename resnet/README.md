# PV Image Clustering

This project groups processed sky images by photovoltaic production and saves a
CNN feature vector for each image.

Default run:

```powershell
python cluster_pv_images.py --model resnet50 --weights imagenet --overwrite
```

By default, rows from hours without useful PV production are discarded before
creating the classes. The automatic rule keeps UTC hours whose maximum
production reaches at least 5% of the global maximum and also drops
`production <= 0`. To force a specific UTC window:

```powershell
python cluster_pv_images.py --active-hours 6-18 --overwrite
```

Outputs are written to `outputs/pv_clusters`:

- `5_classes/very_low`, `5_classes/low`, `5_classes/medium`, `5_classes/high`, `5_classes/very_high`
- `3_classes/low`, `3_classes/medium`, `3_classes/high`
- `assignments.csv` with production and assigned labels per image
- `features_resnet50.npz` with the final feature matrix
- `feature_vectors/5_classes/*.npz` and `feature_vectors/3_classes/*.npz`
  with the same features split by class for the second step
- `summary.json` with counts and production ranges

Alternative VGG16 run:

```powershell
python cluster_pv_images.py --model vgg16 --weights imagenet --output-dir outputs/pv_clusters_vgg16 --overwrite
```

Evaluate whether the saved embeddings separate the PV classes:

```powershell
python evaluate_pv_features.py
```

The evaluation trains a simple linear classifier on the saved features and
writes metrics, classification reports, and confusion matrices under each
model's `evaluation` folder.

Run unsupervised visual clustering on the saved embeddings:

```powershell
python cluster_embeddings_kmeans.py
```

This creates `visual_kmeans_clusters` folders with assignments, metrics, and
confusion matrices comparing visual clusters against the PV-production labels.
It also copies images into
`visual_kmeans_clusters/<n>_clusters/images_by_visual_cluster/<label>`.
