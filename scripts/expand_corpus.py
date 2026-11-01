import os
import shutil

CATEGORIES = ["nodejs", "pandas", "golang", "nvidia_tensorrt-llm", "ros2"]

docs_root = "hf_corpus/corpus/docs"
os.makedirs("corpus", exist_ok=True)

copied = 0
for category in CATEGORIES:
    category_path = os.path.join(docs_root, category)
    if not os.path.isdir(category_path):
        print(f"Skipping missing category: {category}")
        continue

    for root, _, files in os.walk(category_path):
        for filename in files:
            if not filename.endswith(".md"):
                continue
            src = os.path.join(root, filename)
            doc_name = os.path.basename(root)
            dest_name = f"{category}__{doc_name}__{filename}"
            shutil.copy2(src, os.path.join("corpus", dest_name))
            copied += 1

print(f"Copied {copied} additional markdown files into corpus/")