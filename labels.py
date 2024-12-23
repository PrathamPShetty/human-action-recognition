from transformers import ViTForImageClassification, ViTConfig

# Load the model configuration
config = ViTConfig.from_pretrained('google/vit-base-patch16-224-in21k')

# Check if labels exist
if config.id2label:
    labels = config.id2label
else:
    # Fallback to manually generating labels
    labels = {i: f"LABEL_{i}" for i in range(config.num_labels)}

# Print the first 10 labels
for idx in range(10):
    print(f"{idx}: {labels.get(idx, f'LABEL_{idx}')}")
