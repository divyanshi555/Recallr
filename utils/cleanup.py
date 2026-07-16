import os
import shutil

DOWNLOAD_DIR = "downloades"
VECTOR_DB_DIR = "vector_db"


def clean_workspace():
    """
    Prepare a fresh workspace for a new session.

    - Creates the downloads directory if it doesn't exist.
    - Removes all files and subdirectories inside downloads.
    - Deletes the vector database directory if it exists.
    """

    print("🧹 Preparing workspace...")

    # Ensure downloads directory exists
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    # Remove all contents inside downloads
    for item in os.listdir(DOWNLOAD_DIR):
        path = os.path.join(DOWNLOAD_DIR, item)
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.unlink(path)
        except OSError as e:
            print(f"⚠️ Failed to remove '{path}': {e}")

    # Remove previous vector database
    if os.path.isdir(VECTOR_DB_DIR):
        try:
            shutil.rmtree(VECTOR_DB_DIR)
        except OSError as e:
            print(f"⚠️ Failed to remove '{VECTOR_DB_DIR}': {e}")

    print("✅ Workspace is ready.")