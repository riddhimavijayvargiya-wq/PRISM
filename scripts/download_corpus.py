import os
import shutil
from huggingface_hub import snapshot_download

# Step 3: download the dataset
snapshot_download(
    repo_id="Kajaluxan/version-aware-rag-corpus",
    repo_type="dataset",
    local_dir="hf_corpus",
)

# Step 4: pick categories and copy their .md files into your project's corpus/
CATEGORIES = ["python", "pytorch", "kubernetes"]

os.makedirs("corpus", exist_ok=True)
docs_root = "hf_corpus/corpus/docs"

copied = 0
for category in CATEGORIES:
    category_path = os.path.join(docs_root, category)
    if not os.path.isdir(category_path):
        print(f"Skipping missing category: {category}")
        continue

    # Each category has subfolders per document, each holding multiple .md
    # version files — walk everything under it
    for root, _, files in os.walk(category_path):
        for filename in files:
            if not filename.endswith(".md"):
                continue

            src = os.path.join(root, filename)
            # Prefix with category + parent folder name to avoid filename
            # collisions across categories/documents once flattened
            doc_name = os.path.basename(root)
            dest_name = f"{category}__{doc_name}__{filename}"
            dest = os.path.join("corpus", dest_name)

            shutil.copy2(src, dest)
            copied += 1

print(f"Copied {copied} markdown files into corpus/")