import os
import sys
import subprocess
import pickle
import numpy as np
import faiss
import threading
import time
from datetime import datetime
import PyPDF2
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial
import mmap

# Check if SentenceTransformer is installed, if not, install it
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("SentenceTransformer not found. Attempting to install...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])
    from sentence_transformers import SentenceTransformer

# Try to import hf_hub_download, handle potential import errors
try:
    from huggingface_hub import hf_hub_download, cached_download
except ImportError as e:
    print(f"Error importing hf_hub_download: {e}")
    print("Please ensure that huggingface_hub is installed and up to date.")
    hf_hub_download = None

# ╔═════════════════════════════════════════════════════════════════════════╗
# ║                           DRIVE CRAWLER CLASS                           ║
# ╠═════════════════════════════════════════════════════════════════════════╣
# ║                                                                         ║
# ║  This script defines a DriveCrawler class which is responsible for      ║
# ║  crawling a directory, extracting file content, and building an index   ║
# ║  for efficient search functionality. It uses Faiss for indexing and a   ║
# ║  transformer model to create embeddings of the files.                   ║
# ║                                                                         ║
# ╚═════════════════════════════════════════════════════════════════════════╝

class DriveCrawler:
    def __init__(self, root_path, stop_event=None, message_callback=None):
        print(f"DriveCrawler initialized with root_path: {root_path}")  # Debug print
        # INITIALIZATION
        # Initializes DriveCrawler with root path, index file, and model.
        print("Initializing DriveCrawler")
        self.root_path = root_path
        self.index_file = self.generate_index_filename(root_path)
        self.file_paths = []
        self.file_metadata = {}
        self.index = None
        self.last_crawl_time = None
        self.stop_event = stop_event if stop_event else threading.Event()
        self.embedding_model = None
        self.message_callback = message_callback
        self.batch_size = 1000  # Process files in batches of 1000

    def generate_index_filename(self, root_path):
        # GENERATE INDEX FILENAME
        # Generates a unique index filename based on the drive letter
        # and the current date.
        drive_letter = os.path.splitdrive(root_path)[0].rstrip(':')
        date_str = datetime.now().strftime("%d%b%y").upper()
        return f"index_{drive_letter}_{date_str}.pkl"  # Prefix with 'index_'

    def load_index(self):
        # LOAD INDEX
        # Loads an existing index from the filesystem if it exists,
        # otherwise logs that no index is found.
        try:
            if os.path.exists(self.index_file):
                print(f"Loading index from {self.index_file}")
                with open(self.index_file, 'rb') as f:
                    data = pickle.load(f)
                    self.index = faiss.deserialize_index(data['index'])
                    self.file_paths = data['file_paths']
                    self.file_metadata = data['file_metadata']
                    self.last_crawl_time = data.get('last_crawl_time', None)
                print(f"Loaded index with {len(self.file_paths)} files")
                if self.index is not None:
                    print(f"Faiss index contains {self.index.ntotal} embeddings")
                else:
                    print("Warning: Faiss index is None after loading")
            else:
                print(f"No existing index found at {self.index_file}")
        except Exception as e:
            print(f"An error occurred in load_index: {e}")
            import traceback
            traceback.print_exc()

    def crawl_drive(self, callback=None):
        try:
            print("Starting crawl_drive")
            if self.message_callback:
                self.message_callback("Starting crawl_drive")
            start_time = time.time()

            # Define supported file extensions
            supported_extensions = ['.txt', '.md', '.py', '.java', '.csv', '.log', '.pdf', '.xls', '.xlsx']
            # Remove content extraction functions
            extract_functions = {
                '.txt': self.extract_text_from_txt,  # Will modify to skip
                '.md': self.extract_text_from_txt,
                '.py': self.extract_text_from_txt,
                '.java': self.extract_text_from_txt,
                '.csv': self.extract_text_from_txt,
                '.log': self.extract_text_from_txt,
                '.pdf': self.extract_text_from_pdf,
                '.xls': self.extract_text_from_excel,
                '.xlsx': self.extract_text_from_excel
            }

            # Get all supported files
            all_files = self.get_supported_files(supported_extensions)
            total_files = len(all_files)

            print(f"Total supported files to crawl: {total_files}")
            if self.message_callback:
                self.message_callback(f"Total supported files to crawl: {total_files}")

            # Process files in batches without using multiprocessing
            for i in range(0, total_files, self.batch_size):
                if self.stop_event.is_set():
                    print("Crawl stopped.")
                    return

                batch = all_files[i:i+self.batch_size]
                for file in batch:
                    print(f"Processing file: {file}")  # Debug log
                    file_path, file_name = self.process_file((file, supported_extensions, extract_functions))
                    if file_path and file_name:
                        self.file_paths.append(file_path)
                        self.file_metadata[file_path] = {
                            'file_name': file_name,
                        }
                        print(f"Indexed: {file_name}")  # Debug log
                    else:
                        print(f"Failed to process: {file}")  # Debug log

                if callback:
                    callback(min(i + self.batch_size, total_files), total_files)

            self.last_crawl_time = time.time()
            self.rebuild_index()

            if self.index is not None:
                self.save_index()
                print(f"Crawl completed in {time.time() - start_time:.2f} seconds. Indexed {len(self.file_paths)} files.")
                print(f"Faiss index contains {self.index.ntotal} embeddings")
            else:
                print("Warning: Index is None after rebuilding. No data saved.")

            print("Crawl drive completed")
        except Exception as e:
            print(f"An error occurred in crawl_drive: {e}")
            import traceback
            traceback.print_exc()

    def get_supported_files(self, supported_extensions):
        all_files = []
        for root, _, files in os.walk(self.root_path):
            for file in files:
                if os.path.splitext(file)[1].lower() in supported_extensions:
                    all_files.append(os.path.join(root, file))
        return all_files

    def extract_text_from_txt(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    return mm.read().decode('utf-8', errors='ignore')
        except Exception as e:
            print(f"Error extracting text from {file_path}: {e}")
            return ""

    def extract_text_from_pdf(self, file_path):
        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            print(f"Error extracting text from {file_path}: {e}")
            return ""

    def extract_text_from_excel(self, file_path):
        try:
            df = pd.read_excel(file_path, engine='openpyxl')
            return df.to_string()  
        except Exception as e:
            print(f"Error extracting text from {file_path}: {e}")
            return ""

    # INDEX REBUILDING
    # Rebuilds the Faiss index using the file embeddings for fast
    # retrieval during search operations.
    def rebuild_index(self):
        try:
            print("Starting rebuild_index")
            batch_size = 64
            embeddings = []
            file_paths = self.file_paths.copy()

            for i in range(0, len(file_paths), batch_size):
                batch_files = file_paths[i:i + batch_size]
                batch_file_names = [self.file_metadata[file]['file_name'] for file in batch_files]
                batch_embeddings = self.get_embedding_model().encode(
                    batch_file_names, 
                    batch_size=batch_size, 
                    show_progress_bar=True
                )
                
                for file, embedding in zip(batch_files, batch_embeddings):
                    self.file_metadata[file]['embedding'] = embedding
                    embeddings.append(embedding)
                
                if self.stop_event.is_set():
                    print("Index rebuilding stopped.")
                    return

            embeddings = np.array(embeddings).astype('float32')
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(embeddings)
            print(f"Index rebuilt successfully with {self.index.ntotal} embeddings")
        except Exception as e:
            print(f"An error occurred in rebuild_index: {e}")
            import traceback
            traceback.print_exc()

    # SEARCH FUNCTIONALITY
    # Searches the indexed files based on a query vector and returns
    # the top k results.
    def search(self, query, _selected_dbs=None, k=10):
        try:
            print("Starting search")
            if not self.file_paths:
                print("No files have been indexed yet. Please crawl a drive first.")
                return [], ""

            # Use get_embedding_model() instead of self.model
            query_vector = self.get_embedding_model().encode([query], convert_to_numpy=True)[0]
            k = min(int(k), len(self.file_paths))

            distances, indices = self.index.search(np.array([query_vector]).astype('float32'), k)

            print(f"Search results (Distance, Index):")
            for distance, index in zip(distances[0], indices[0]):
                print(f"{distance:.4f}, {index}")

            results = []
            for distance, index in zip(distances[0], indices[0]):
                file_path = self.file_paths[index]
                file_name = self.file_metadata[file_path]['file_name']
                
                # Check if query is in file name (case-insensitive)
                if query.lower() in file_name.lower():
                    result = {
                        "file_path": file_path,
                        "file_name": file_name,
                        "distance": float(distance),
                    }
                    results.append(result)

            # Sort results by distance
            results.sort(key=lambda x: x['distance'])

            print("Search completed with results:")
            for res in results:
                print(f"File: {res['file_path']} - Distance: {res['distance']:.4f}")

            return results, query
        except Exception as e:
            print(f"An error occurred in search: {e}")
            import traceback
            traceback.print_exc()
            return [], ""

    # EMBEDDING MODEL MANAGEMENT
    # Lazy loads the embedding model to reduce initialization time.
    def get_embedding_model(self):
        print("Entering get_embedding_model method")  # Debug print
        if self.embedding_model is None:
            print("Initializing embedding model...")
            if SentenceTransformer is None:
                raise ImportError("SentenceTransformer is not available. Please install sentence-transformers.")
            
            model_name = 'sentence-transformers/all-MiniLM-L6-v2'
            try:
                # Try to load the model directly
                self.embedding_model = SentenceTransformer(model_name)
                print("Embedding model loaded successfully.")
            except ImportError as e:
                print(f"Import error while loading SentenceTransformer: {e}")
                raise
            except Exception as e:
                print(f"Error loading model directly: {e}")
                if hf_hub_download is None:
                    raise ImportError("hf_hub_download is not available. Please install huggingface_hub.")
                try:
                    # If direct loading fails, try using hf_hub_download
                    model_path = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
                    self.embedding_model = SentenceTransformer(model_path)
                    print("Embedding model loaded successfully using hf_hub_download.")
                except Exception as e:
                    print(f"Error loading model using hf_hub_download: {e}")
                    raise
        else:
            print("Using existing embedding model.")
        
        print(f"Returning embedding model: {self.embedding_model}")  # Debug print
        return self.embedding_model

    # PLACEHOLDER FUNCTIONS
    # Placeholder for future implementations (e.g., brute-force
    # search or clearing conversation history).
    def brute_force_search(self, query_vector, k):
        pass

    def clear_conversation(self):
        """
        Clears any conversation history or related data.
        Currently a placeholder. Implement if necessary.
        """
        pass

    def save_index(self):
        try:
            print(f"Attempting to save index to {self.index_file}")
            print(f"Current working directory: {os.getcwd()}")
            
            # Ensure the directory exists
            index_dir = os.path.dirname(self.index_file)
            if not index_dir:
                index_dir = os.getcwd()  # Use current working directory if no directory is specified
            os.makedirs(index_dir, exist_ok=True)
            
            full_path = os.path.join(index_dir, os.path.basename(self.index_file))
            
            with open(full_path, 'wb') as f:
                pickle.dump({
                    'index': faiss.serialize_index(self.index),
                    'file_paths': self.file_paths,
                    'file_metadata': self.file_metadata,
                    'last_crawl_time': self.last_crawl_time
                }, f)
            print(f"Index saved successfully to {full_path}")
            
            # List files in the directory after saving
            print("Files in the directory after saving:")
            for file in os.listdir(index_dir):
                print(f"  {file}")
        except Exception as e:
            print(f"Error saving index: {e}")
            import traceback
            traceback.print_exc()

    def process_file(self, args):
        file_path, supported_extensions, extract_functions = args
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension in supported_extensions:
                # Skip content extraction
                file_name = os.path.basename(file_path)
                return file_path, file_name
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
        return None, None

    def index_file(self, file_path):
        try:
            file_name = os.path.basename(file_path)
            print(f"Indexing file: {file_name}")  # Debug log
            
            # Create embeddings for both file name and content
            file_name_embedding = self.get_embedding_model().encode([file_name])[0]
            
            # Read file content (limit to first 1000 characters for large files)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read(1000)
            
            content_embedding = self.get_embedding_model().encode([content])[0]
            
            # Combine embeddings (you can adjust the weights)
            combined_embedding = 0.5 * file_name_embedding + 0.5 * content_embedding
            
            self.index.add(np.array([combined_embedding]))
            self.file_paths.append(file_path)
            self.file_metadata[file_path] = {
                'file_name': file_name,
            }
            print(f"Successfully indexed: {file_name}")  # Debug log
        except Exception as e:
            print(f"Error indexing file {file_path}: {e}")

    def search(self, query, _selected_dbs=None, k=10):
        try:
            print(f"Starting search for query: '{query}'")
            if not self.file_paths:
                print("No files have been indexed yet. Please crawl a drive first.")
                return [], ""

            embedding_model = self.get_embedding_model()
            query_vector = embedding_model.encode([query], convert_to_numpy=True)[0]
            k = min(int(k), len(self.file_paths))

            distances, indices = self.index.search(np.array([query_vector]).astype('float32'), k)

            print(f"Search results (Distance, Index, File Name):")
            results = []
            for distance, index in zip(distances[0], indices[0]):
                file_path = self.file_paths[index]
                file_name = self.file_metadata[file_path]['file_name']
                print(f"{distance:.4f}, {index}, {file_name}")
                
                # More lenient filtering: check if any word from the query is in the file name
                query_words = query.lower().split()
                if any(word in file_name.lower() for word in query_words):
                    result = {
                        "file_path": file_path,
                        "file_name": file_name,
                        "distance": float(distance),
                    }
                    results.append(result)
                else:
                    print(f"Filtered out: {file_name}")

            # Sort results by distance
            results.sort(key=lambda x: x['distance'])

            print(f"Search completed with {len(results)} results:")
            for res in results:
                print(f"File: {res['file_name']} - Distance: {res['distance']:.4f}")

            return results, query
        except Exception as e:
            print(f"An error occurred in search: {e}")
            import traceback
            traceback.print_exc()
            return [], ""



