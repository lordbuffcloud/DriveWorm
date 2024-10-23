import os
import json
import pickle
import gzip
from datetime import datetime

class IndexManager:
    def __init__(self, index_dir='indexes'):
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)
        self.metadata_file = os.path.join(index_dir, 'index_metadata.json')
        self.load_metadata()

    def load_metadata(self):
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

    def save_metadata(self):
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def create_index(self, name, root_path):
        index_file = os.path.join(self.index_dir, f'{name}.pkl.gz')
        self.metadata[name] = {
            'creation_date': datetime.now().isoformat(),
            'last_update': datetime.now().isoformat(),
            'root_path': root_path,
            'file_count': 0,
            'version': 1
        }
        self.save_metadata()
        return index_file

    def update_index(self, name, index_data, file_count):
        if name not in self.metadata:
            raise ValueError(f"Index '{name}' does not exist.")
        
        index_file = os.path.join(self.index_dir, f'{name}.pkl.gz')
        with gzip.open(index_file, 'wb') as f:
            pickle.dump(index_data, f)
        
        self.metadata[name]['last_update'] = datetime.now().isoformat()
        self.metadata[name]['file_count'] = file_count
        self.metadata[name]['version'] += 1
        self.save_metadata()

    def load_index(self, name):
        if name not in self.metadata:
            raise ValueError(f"Index '{name}' does not exist.")
        
        index_file = os.path.join(self.index_dir, f'{name}.pkl.gz')
        with gzip.open(index_file, 'rb') as f:
            return pickle.load(f)

    def delete_index(self, name):
        if name not in self.metadata:
            raise ValueError(f"Index '{name}' does not exist.")
        
        index_file = os.path.join(self.index_dir, f'{name}.pkl.gz')
        os.remove(index_file)
        del self.metadata[name]
        self.save_metadata()

    def list_indexes(self):
        return list(self.metadata.keys())

    def get_index_info(self, name):
        return self.metadata.get(name, None)

# Usage in DriveCrawler
class DriveCrawler:
    def __init__(self, root_path, index_name, stop_event=None, message_callback=None):
        self.root_path = root_path
        self.index_manager = IndexManager()
        self.index_name = index_name
        self.index_file = self.index_manager.create_index(index_name, root_path)
        # ... rest of the initialization ...

    def save_index(self):
        index_data = {
            'index': faiss.serialize_index(self.index),
            'file_paths': self.file_paths,
            'file_metadata': self.file_metadata,
            'last_crawl_time': self.last_crawl_time
        }
        self.index_manager.update_index(self.index_name, index_data, len(self.file_paths))

    def load_index(self):
        try:
            data = self.index_manager.load_index(self.index_name)
            self.index = faiss.deserialize_index(data['index'])
            self.file_paths = data['file_paths']
            self.file_metadata = data['file_metadata']
            self.last_crawl_time = data.get('last_crawl_time', None)
            print(f"Loaded index with {len(self.file_paths)} files")
        except Exception as e:
            print(f"Error loading index: {e}")

# In the main application
class DriveWormApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.index_manager = IndexManager()
        # ... rest of the initialization ...

    def create_index_management_ui(self):
        # Create a new window or frame for index management
        index_window = tk.Toplevel(self)
        index_window.title("Index Management")

        # List of indexes
        self.index_listbox = tk.Listbox(index_window)
        self.index_listbox.pack(pady=10)
        self.update_index_list()

        # Buttons for index operations
        tk.Button(index_window, text="Create New Index", command=self.create_new_index).pack()
        tk.Button(index_window, text="Delete Selected Index", command=self.delete_selected_index).pack()
        tk.Button(index_window, text="View Index Info", command=self.view_index_info).pack()

    def update_index_list(self):
        self.index_listbox.delete(0, tk.END)
        for index in self.index_manager.list_indexes():
            self.index_listbox.insert(tk.END, index)

    def create_new_index(self):
        # Implement logic to create a new index
        pass

    def delete_selected_index(self):
        # Implement logic to delete the selected index
        pass

    def view_index_info(self):
        # Implement logic to display information about the selected index
        pass
