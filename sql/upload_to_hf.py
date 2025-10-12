#!/usr/bin/env python3
"""
Script to upload groundSQL dataset to Hugging Face Hub.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, login
import json

def validate_dataset_file(file_path):
    """Validate that the dataset file exists and is properly formatted."""
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    # Check if it's a valid JSONL file by reading a few lines
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 3:  # Check first 3 lines
                    break
                json.loads(line.strip())
        print(f"✓ Dataset file validated: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSONL format in {file_path}: {e}")

def login_to_huggingface():
    """Login to Hugging Face Hub and return the token."""
    # Check for HF_TOKEN
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("❌ Error: HF_TOKEN environment variable not set.")
        print("Please set it with: export HF_TOKEN=your_token_here")
        print("On Windows: set HF_TOKEN=your_token_here")
        print("On Unix/Mac: export HF_TOKEN=your_token_here")
        sys.exit(1)
    
    try:
        # Login to Hugging Face
        print("🔑 Logging into Hugging Face...")
        login(token=hf_token)
        
        # Verify login by testing API access
        api = HfApi(token=hf_token)
        user_info = api.whoami()
        username = user_info['name']
        
        print(f"✅ Successfully logged in to Hugging Face as: {username}")
        return hf_token, api, username
        
    except Exception as e:
        print(f"❌ Failed to login to Hugging Face: {e}")
        print("Please check your HF_TOKEN is valid and has the necessary permissions.")
        sys.exit(1)

def upload_dataset():
    """Upload the groundSQL dataset to Hugging Face Hub."""
    
    # Login first
    hf_token, api, username = login_to_huggingface()
    
    # Dataset configuration (configurable via environment variables)
    dataset_name = os.getenv('DATASET_NAME', 'groundSQL')
    dataset_source = os.getenv('DATASET_SOURCE', 'result/dataset.jsonl')
    dataset_file = Path(dataset_source)
    repo_id = f"{username}/{dataset_name}"
    
    try:
        # Validate dataset file
        validate_dataset_file(dataset_file)
        
        print(f"📤 Uploading dataset '{dataset_name}' to {repo_id}...")
        print(f"📂 Source file: {dataset_file}")
        
        # Create repository if it doesn't exist
        try:
            api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                exist_ok=True,
                private=False
            )
            print(f"✓ Repository created/verified: {repo_id}")
        except Exception as e:
            print(f"Warning: Could not create repository: {e}")
        
        # Upload the dataset file
        api.upload_file(
            path_or_fileobj=str(dataset_file),
            path_in_repo="dataset.jsonl",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload groundSQL dataset"
        )
        
        # Create a README.md file
        readme_content = """---
license: mit
task_categories:
- text2sql
- question-answering
language:
- en
tags:
- sql
- database
- text-to-sql
- code-generation
size_categories:
- 1K<n<10K
---

# groundSQL Dataset

This dataset contains SQL queries with their corresponding solutions and test cases for text-to-SQL tasks.

## Dataset Structure

The dataset is provided in JSONL format where each line contains:

- `id`: Unique identifier for the query
- `message`: Contains the user's request and database schema
- `groundtruth`: List of test cases to validate the SQL solution
- `solution`: The correct SQL query
- `dataset`: Source dataset category
- `difficulty`: Query difficulty level (easy/medium/hard)
- `database_used`: The database file used for this query

## Usage

```python
import json

# Load the dataset
with open('dataset.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

# Example: Get all medium-level queries
medium_queries = [item for item in data if item['difficulty'] == 'medium']
```

## Citation

If you use this dataset, please cite appropriately.
"""
        
        # Upload README
        api.upload_file(
            path_or_fileobj=readme_content.encode('utf-8'),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add README.md"
        )
        
        print(f"✅ Successfully uploaded {dataset_name} dataset!")
        print(f"🔗 Dataset URL: https://huggingface.co/datasets/{repo_id}")
        
        # Print dataset statistics
        with open(dataset_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"📊 Dataset contains {len(lines)} SQL queries")
            
        # Count by difficulty
        difficulties = {}
        for line in lines:
            data = json.loads(line.strip())
            diff = data.get('difficulty', 'unknown')
            difficulties[diff] = difficulties.get(diff, 0) + 1
        
        print("📈 Difficulty distribution:")
        for diff, count in sorted(difficulties.items()):
            print(f"  - {diff}: {count}")
            
    except Exception as e:
        print(f"❌ Error uploading dataset: {e}")
        sys.exit(1)

if __name__ == "__main__":
    upload_dataset()
