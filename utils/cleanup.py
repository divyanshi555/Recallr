import os
import shutil

from core.vector_store import clear_vector_store

DOWNLOAD_DIR = "downloads"


'''
 Function:Prepare a fresh workspace for a new session.
 - Creates the downloads directory if it doesn't exist.
 - Clears the vector database's transcript collection
'''
def clean_workspace():

    print("🧹 Preparing workspace...")
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    for item in os.listdir(DOWNLOAD_DIR):
        path = os.path.join(DOWNLOAD_DIR, item)
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.unlink(path)
        except OSError as e:
            print(f"⚠️ Failed to remove '{path}': {e}")

    clear_vector_store()

    print("✅ Workspace is ready.")