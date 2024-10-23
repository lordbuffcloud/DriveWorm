import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
from PIL import Image, ImageTk
import threading
import datetime
import os
import subprocess
import torch
from pathlib import Path
import time
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from functools import lru_cache
import warnings
import traceback

# Add these lines near the top of the file, after the imports
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.file_download")

# Attempt to import transformers, handle potential import errors
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
except ImportError as e:
    print(f"Error importing transformers: {e}")
    print("Some AI features may be unavailable.")
    AutoTokenizer = AutoModelForCausalLM = AutoConfig = None

# Attempt to import DriveCrawler, handle potential import error
try:
    from drive_crawler import DriveCrawler
except ImportError as e:
    print(f"Error importing DriveCrawler: {e}")
    print("Please ensure that all required libraries are installed and up to date.")
    DriveCrawler = None

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                             DRIVE WORM APP CLASS                           ║
# ╠════════════════════════════════════════════════════════════════════════════╣
# ║ This script defines a DriveWormApp class, which is a GUI-based application ║
# ║ built using Tkinter. It enables users to search, chat, and interact with   ║
# ║ various functionalities like drive crawling and AI-based conversation.     ║
# ╚════════════════════════════════════════════════════════════════════════════╝

class ModernWidget(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        
        # ╔════════════════════════════════════════════════════════════════╗
        # ║ MODERN WIDGET INIT                                              ║
        # ╠═════════════════════════════════════════════════════════════════╣
        # ║ Inherits from ttk.Frame and sets a modern style for consistent  ║
        # ║ UI appearance across all frames used in the DriveWormApp.       ║
        # ╚═════════════════════════════════════════════════════════════════╝
        super().__init__(parent, *args, **kwargs)
        self.configure(style='Modern.TFrame')

class DriveWormApp(tk.Tk):
    def __init__(self):
        try:
            super().__init__()
            
            # Set the window title
            self.title("DriveWorm")
            self.geometry("1200x800")

            # Set the window icon
            self.set_icon()

            if DriveCrawler is None:
                raise ImportError("Failed to import DriveCrawler.")

            self.attributes('-alpha', .8)  # Set window to semi-transparent

            # ╔═ LOAD AI MODEL ═══════════════════════════════════════════════╗
            # Initialize the Llama 2 model and tokenizer
            self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            print(f"Loading model: {self.model_name}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, 
                    trust_remote_code=True, 
                    cache_dir='./model_cache'
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    cache_dir='./model_cache'
                )

                # Set pad_token to a new token if pad_token is not set
                if self.tokenizer.pad_token is None:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    self.model.resize_token_embeddings(len(self.tokenizer))

                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(f"Using device: {self.device}")
                self.model.to(self.device)
                print("Model loaded successfully")
            except Exception as e:
                print(f"Error loading AI model: {e}")
                self.model = None
                self.tokenizer = None
                messagebox.showwarning(
                    "Model Loading Error", 
                    f"Failed to load the AI model. Some features may be unavailable.\nError: {e}"
            )

            def get_ai_response(self, user_message):
                try:
                    print("Starting get_ai_response")
                    if self.model is None or self.tokenizer is None:
                        print("Error: Model or tokenizer is not initialized")
                        return self.update_conversation("I'm sorry, the AI model is not ready. Please try again later.")

                    self.after(0, self.start_chat_worm_animation)  # Start worm animation

                    # Optimize prompt generation
                    relevant_contents = self.get_relevant_contents()
                    prompt = self.create_prompt(user_message, relevant_contents)

                    print(f"Generated prompt: {prompt[:500]}...")  # Print first 500 characters of prompt

                    try:
                        print("Generating AI response...")
                        start_time = time.time()

                        # Prepare the input for the model
                        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

                        # Generate response with optimized settings
                        output = self.model.generate(
                            input_ids,
                            max_length=1024,  # Adjust this to control the response length
                            num_beams=3,  # Beam search for more coherent responses
                            do_sample=True,
                            temperature=0.7  # Adjust temperature to control randomness
                        )
                        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
                        generation_time = time.time() - start_time
                        print(f"Response generated in {generation_time:.2f} seconds")
                        print(f"Response preview: {response[:100]}...")  # Print first 100 chars

                        # Schedule the GUI update on the main thread
                        self.after(0, lambda: self.update_conversation(response))

                    except TimeoutError:
                        print("Main model timed out, attempting fallback response")
                        ai_response = self.generate_fallback_response(prompt)
                        self.after(0, lambda: self.update_conversation(ai_response))

                    except Exception as e:
                        error_message = f"An error occurred while generating AI response: {e}"
                        print(error_message)
                        traceback.print_exc()
                        self.after(0, lambda: self.update_conversation("I'm sorry, I couldn't process your request at the moment."))

                except Exception as e:
                    error_message = f"An error occurred in get_ai_response: {e}"
                    print(error_message)
                    traceback.print_exc()
                    self.after(0, lambda: self.update_conversation("I apologize, but an error occurred while processing your request."))


            
            # ╔═ INITIALIZE VARIABLES ══════════════════════════════════════════╗
            # Initialize worm_loading
            self.worm_loading = None
            
            # Initialize search_results and its lock to prevent AttributeError
            self.search_results = []
            self.search_results_lock = threading.Lock()
            
            # Initialize stop_event for managing long-running operations
            self.stop_event = threading.Event()
            
            self.setup_styles()
            self.create_title_bar()
            self.create_canvas()
            self.create_widgets()  # Make sure this is called before creating tooltips
            
            # Load worm frames before creating loading screen
            self.load_worm_frames()
            
            # Create the initial loading screen before loading the main UI
            self.create_loading_screen()
            
            # Start preloading indexes using a background thread
            threading.Thread(target=self.preload_indexes, daemon=True).start()
            
            # Dictionary to hold multiple DriveCrawler instances
            self.crawlers = {}
            
            # Load the logo image for chat responses
            try:
                self.chat_logo_image = Image.open(os.path.join(os.path.dirname(__file__), "images", "dw.png"))
                self.chat_logo_image = self.chat_logo_image.resize((20, 20), Image.LANCZOS)  # Resize for chat
                self.chat_logo_photo = ImageTk.PhotoImage(self.chat_logo_image)
            except FileNotFoundError:
                messagebox.showerror("Error", "Logo image 'dw.png' not found in the 'images' directory.")
                self.destroy()
                return

            # Add tooltips to various widgets
            self.create_tooltip(self.drive_entry, "Enter the path of the drive you want to crawl")
            self.create_tooltip(self.browse_button, "Click to browse and select a drive")
            self.create_tooltip(self.crawl_button, "Start crawling the selected drive")
            self.create_tooltip(self.query_entry, "Enter your search query here")
            self.create_tooltip(self.search_button, "Click to search the indexed files")
            self.create_tooltip(self.results_listbox, "Search results will appear here. Double-click to open a file.")
            self.create_tooltip(self.user_input, "Type your message to the AI here")
            self.create_tooltip(self.send_button, "Send your message to the AI")

            # Add a help button
            self.help_button = ttk.Button(self.button_frame, text="?", command=self.show_help)
            self.help_button.pack(side=tk.RIGHT, padx=5)
            self.create_tooltip(self.help_button, "Click for help on how to use DriveWorm")

        except Exception as e:
            print(f"Error during DriveWormApp initialization: {e}")
            messagebox.showerror("Initialization Error", f"An error occurred during application startup: {e}")
            self.destroy()
            raise

    def set_icon(self):
        try:
            icon_path = os.path.join(os.path.dirname(__file__), "images", "dw_icon.ico")
            self.iconbitmap(icon_path)
            print(f"Successfully set icon using iconbitmap: {icon_path}")
        except tk.TclError as e:
            print(f"Error setting icon with iconbitmap: {e}")
            try:
                # Fallback to using PhotoImage for the icon
                icon_image = tk.PhotoImage(file=os.path.join(os.path.dirname(__file__), "images", "dw.png"))
                self.tk.call('wm', 'iconphoto', self._w, icon_image)
                print("Successfully set icon using PhotoImage")
            except Exception as e:
                print(f"Error setting icon with PhotoImage: {e}")

    # ╔═════════════════════════════════════════════════════════════════╗
    # ║ LOADING SCREEN                                                  ║
    # ╠═════════════════════════════════════════════════════════════════╣
    # ║ Handles creating the loading screen, including logo and worm    ║
    # ║ animation for when the app is initializing or loading data.     ║
    # ╚═════════════════════════════════════════════════════════════════╝
    def create_loading_screen(self):
        # Create a loading screen widget
        self.loading_frame = ModernWidget(self)
        self.loading_frame.pack(fill=tk.BOTH, expand=True)
        
        # Set the loading screen to be topmost
        self.attributes('-topmost', True)
        
        # Set the icon for the loading screen
        try:
            icon_path = os.path.join(os.path.dirname(__file__), "images", "dw_icon.ico")
            self.iconbitmap(icon_path)
        except tk.TclError:
            print(f"Warning: Could not load icon from {icon_path} using iconbitmap")
            try:
                # Fallback to using PhotoImage for the icon
                icon_image = tk.PhotoImage(file=os.path.join(os.path.dirname(__file__), "images", "dw.png"))
                self.tk.call('wm', 'iconphoto', self._w, icon_image)
            except Exception as e:
                print(f"Warning: Could not load icon using PhotoImage: {e}")
        
        # Load the logo image
        try:
            logo_image = Image.open(os.path.join(os.path.dirname(__file__), "images", "dw.png"))
            desired_height = 100  # Adjust logo size as needed
            aspect_ratio = logo_image.width / logo_image.height
            desired_width = int(desired_height * aspect_ratio)
            logo_image = logo_image.resize((desired_width, desired_height), Image.LANCZOS)
            self.loading_logo_photo = ImageTk.PhotoImage(logo_image)
        except FileNotFoundError:
            messagebox.showerror("Error", "Logo image 'dw.png' not found in the 'images' directory.")
            self.destroy()
            return
        
        # Display logo on loading screen
        logo_label = tk.Label(self.loading_frame, image=self.loading_logo_photo, bg='#333333')
        logo_label.pack(pady=20)
        
        # App name label
        app_name_label = ttk.Label(self.loading_frame, text="DriveWorm", style='Modern.TLabel', font=('Segoe UI', 24, 'bold'))
        app_name_label.pack(pady=10)
        
        # Status label
        self.loading_status = ttk.Label(self.loading_frame, text="Loading...", style='Modern.TLabel')
        self.loading_status.pack(pady=10)
        
        # Worm animation
        self.loading_canvas = tk.Canvas(self.loading_frame, width=400, height=120, bg='#333333', highlightthickness=0)
        self.loading_canvas.pack(pady=10)
        self.animate_loading_worm()

        # Version label
        version_label = ttk.Label(self.loading_frame, text="v1.0", style='Modern.TLabel', font=('Segoe UI', 10))
        version_label.pack(side=tk.BOTTOM, pady=10)

        # Center the loading screen on the display
        self.center_window()

    def center_window(self):
        # Get the screen width and height
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        # Calculate the x and y coordinates for the window
        x = (screen_width - self.winfo_width()) // 2
        y = (screen_height - self.winfo_height()) // 2

        # Set the position of the window
        self.geometry(f'+{x}+{y}')

    # ╔═══════🐛══════════════════════════════════════════════════════════════🐛
    # ║ PRELOAD INDEXES                                                            
    # ╠══════🐛═══════════════════════════════════════════════════════════════🐛
    # ║ Preloads drive indexes to speed up searching and interaction.              
    # ╚═══════🐛══════════════════════════════════════════════════════════🐛
    def preload_indexes(self):
        try:
            print("Starting preload_indexes")
            self.loading_start_time = time.time()
            self.crawler = DriveCrawler(root_path='D:\\', stop_event=self.stop_event)
            self.crawler.load_index()
            print("Finished loading index")
            
            elapsed = time.time() - self.loading_start_time
            # Ensure loading takes at least 10 seconds to show animation
            if elapsed < 10:
                delay_ms = int((10 - elapsed) * 1000)
                self.after(delay_ms, self.finish_loading)
            else:
                self.after(0, self.finish_loading)
        except Exception as e:
            print(f"An error occurred in preload_indexes: {e}")
            import traceback
            traceback.print_exc()

    # ╔═════════════════════════════════════════════════════════════════╗
    # ║ WORM ANIMATION START                                            ║
    # ╠═════════════════════════════════════════════════════════════════╣
    # ║ Starts the worm animation used as a loading indicator           ║
    # ║ the things, giving visual feedback to the user.                 ║
    # ╚═════════════════════════════════════════════════════════════════╝
    
    def animate_loading_worm(self):
        try:
            if not hasattr(self, 'worm_frames') or not self.worm_frames:
                print("Worm frames not loaded. Cannot animate worm.")
                return
            
            # Create worm image on canvas if it doesn't exist
            if self.worm_loading is None:
                print("Creating worm_loading image on the canvas.")
                self.worm_loading = self.loading_canvas.create_image(0, 20, image=self.worm_frames[self.current_frame], anchor='nw')
            
            # Update to the next frame
            self.current_frame = (self.current_frame + 1) % len(self.worm_frames)
            self.loading_canvas.itemconfig(self.worm_loading, image=self.worm_frames[self.current_frame])
            print(f"Displaying frame {self.current_frame}")
            
            # Move the worm across the canvas
            self.loading_canvas.move(self.worm_loading, 5, 0)
            print(f"Moved worm to position: {self.loading_canvas.coords(self.worm_loading)}")
            
            # Get current position and reset if needed
            worm_pos = self.loading_canvas.coords(self.worm_loading)
            canvas_width = self.loading_canvas.winfo_width()
            if worm_pos[0] > canvas_width:
                print("Resetting worm position to start.")
                self.loading_canvas.coords(self.worm_loading, -80, 20)  # Reset position
            
            # Schedule the next frame
            self.loading_animation = self.after(150, self.animate_loading_worm)
            print("Scheduled next worm animation frame.")
        except Exception as e:
            print(f"An error occurred in animate_loading_worm: {e}")
            import traceback
            traceback.print_exc()

    # ╔═════════════════════════════════════════════════════════════════╗
    # ║ FINISH LOADING                                                  ║
    # ╠═════════════════════════════════════════════════════════════════╣
    # ║ Finalizes the loading screen and initializes the main GUI once  ║
    # ║ loading is complete.                                            ║
    # ╚═════════════════════════════════════════════════════════════════╝
    
    def finish_loading(self):
        # Stop loading animation
        if hasattr(self, 'loading_animation'):
            self.after_cancel(self.loading_animation)
        if self.worm_loading is not None:
            self.loading_canvas.delete(self.worm_loading)
            self.worm_loading = None
        
        # Remove topmost attribute
        self.attributes('-topmost', False)
        
        # Destroy loading frame
        self.loading_frame.destroy()
        
        # Initialize main GUI
        self.create_main_gui()
    
    # ╔═════════════════════════════════════════════════════════════════╗
    # ║ MAIN GUI CREATION                                               ║
    # ╠═════════════════════════════════════════════════════════════════╣
    # ║ Handles creation of the main GUI, including title bar, widgets, ║
    # ║ and overall layout setup.                                       ║
    # ╚════════════════════════════════════════════════════════════════╝
    
    def create_main_gui(self):
        self.setup_styles()
        self.create_title_bar()
        self.create_canvas()  # Initialize the canvas first
        self.create_widgets()
        self.load_worm_frames()
        self.load_databases()  # Call this method to populate the listbox

    # ╔═════════════════════════════════════════════════════════════════╗
    # ║ SETUP STYLES                                                    ║
    # ╠═════════════════════════════════════════════════════════════════╣
    # ║ Configures styles for Tkinter widgets to maintain consistency   ║
    # ║ in the user interface.                                          ║
    # ╚═════════════════════════════════════════════════════════════════╝
    
    def setup_styles(self):
        style = ttk.Style(self)
        style.theme_use('clam')

        # Define colors for consistent styling
        bg_color = '#333333'
        fg_color = '#FFFFFF'
        accent_color = '#80CFFF'
        secondary_color = '#A0DFFF'
        box_bg_color = '#222222'

        self.configure(bg=bg_color)

        # Apply styles to various widget types
        style.configure('Modern.TFrame', background=bg_color)
        style.configure('Modern.TLabel', background=bg_color, foreground=fg_color, font=('Segoe UI', 10))
        style.configure('Modern.TButton', background=accent_color, foreground=bg_color, font=('Segoe UI', 10), borderwidth=0)
        style.map('Modern.TButton', background=[('active', secondary_color)])
        style.configure('Modern.TEntry', fieldbackground=bg_color, foreground=fg_color, font=('Segoe UI', 10), borderwidth=0)
        style.configure('Modern.Horizontal.TProgressbar', background=accent_color, troughcolor=bg_color)
        style.configure('Modern.TNotebook', background=bg_color, borderwidth=0)
        style.configure('Modern.TNotebook.Tab', background=bg_color, foreground=fg_color, padding=[10, 5], font=('Segoe UI', 10))
        style.map('Modern.TNotebook.Tab', background=[('selected', accent_color)], foreground=[('selected', bg_color)])
        style.configure('Modern.Vertical.TScrollbar', troughcolor=bg_color, background=accent_color, arrowcolor=fg_color)

    # ╔═══════════════════════════════════════════════════════════════╗
    # ║ CREATE TITLE BAR                                              ║
    # ╠═══════════════════════════════════════════════════════════════╣
    # ║ Creates a custom title bar with logo, minimize, close, and    ║
    # ║ always-on-top functionality. Also makes the window draggable. ║
    # ╚═══════════════════════════════════════════════════════════════╝
    
    def create_title_bar(self):
        title_bar = tk.Frame(self, bg='#222222', relief='flat', bd=0)
        title_bar.pack(fill=tk.X)

        # Load and display the logo image
        try:
            logo_image = Image.open(os.path.join(os.path.dirname(__file__), "images", "dw.png"))
        except FileNotFoundError:
            messagebox.showerror("Error", "Logo image 'dw.png' not found in the 'images' directory.")
            self.destroy()
            return

        # Resize the image for the title bar
        desired_height = 30  # Adjust this value to fit your title bar
        aspect_ratio = logo_image.width / logo_image.height
        desired_width = int(desired_height * aspect_ratio)
        logo_image = logo_image.resize((desired_width, desired_height), Image.LANCZOS)

        self.logo_photo = ImageTk.PhotoImage(logo_image)

        # Display the logo in the title bar
        logo_label = tk.Label(title_bar, image=self.logo_photo, bg='#222222')
        logo_label.pack(side=tk.LEFT, padx=10)

        # Add close button
        close_button = tk.Button(title_bar, text="✕", bg='#222222', fg='#888888', 
                                 font=('Segoe UI', 12), bd=0, 
                                 activebackground='#333333', activeforeground='#FFFFFF',
                                 command=self.on_closing)
        close_button.pack(side=tk.RIGHT, padx=5)

        # Add minimize button
        minimize_button = tk.Button(title_bar, text="—", bg='#222222', fg='#888888', 
                                    font=('Segoe UI', 12), bd=0, 
                                    activebackground='#333333', activeforeground='#FFFFFF',
                                    command=self.iconify)
        minimize_button.pack(side=tk.RIGHT, padx=5)

        # Add always-on-top toggle button
        self.always_on_top_button = tk.Button(title_bar, text="📌", bg='#222222', fg='#888888', 
                                              font=('Segoe UI', 12), bd=0, 
                                              activebackground='#333333', activeforeground='#FFFFFF',
                                              command=self.toggle_always_on_top)
        self.always_on_top_button.pack(side=tk.RIGHT, padx=5)

        # Make the title bar draggable
        title_bar.bind('<Button-1>', self.get_pos)
        title_bar.bind('<B1-Motion>', self.move_window)

        # Remove the default title bar
        self.overrideredirect(True)

    def get_pos(self, event):
        # Store initial click position for dragging window
        self.xwin = event.x
        self.ywin = event.y

    def move_window(self, event):
        # Calculate and set new window position
        self.geometry(f'+{event.x_root - self.xwin}+{event.y_root - self.ywin}')

    # ╔════════════════════════════════════════════════════════════════╗
    # ║ CANVAS CREATION                                                ║
    # ╠════════════════════════════════════════════════════════════════╣
    # ║ Creates the main canvas to host the worm animation.            ║
    # ╚════════════════════════════════════════════════════════════════╝
    
    def create_canvas(self):
        animation_frame = ModernWidget(self)
        animation_frame.pack(fill=tk.X, pady=(10, 0))
        self.canvas = tk.Canvas(animation_frame, width=1160, height=120, bg='#333333', highlightthickness=0)  # Increased height from 100 to 120
        self.canvas.pack()
        # Initialize worm as None
        self.worm = None

    # ╔═════════════════════════════════════════════════════════════════╗
    # ║ MAIN WIDGET CREATION                                            ║
    # ╠═════════════════════════════════════════════════════════════════╣
    # ║ Creates the primary widgets used in the main GUI, including     ║
    # ║ tabs for drive selection, searching, and chatting.              ║
    # ╚═════════════════════════════════════════════════════════════════╝
    def create_widgets(self):
        main_frame = ModernWidget(self)
        main_frame.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)

        # Notebook for tabs
        notebook = ttk.Notebook(main_frame, style='Modern.TNotebook')
        notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # Drive selection and scanning tab
        drive_tab = ModernWidget(notebook)
        notebook.add(drive_tab, text='Drives')

        # ╔═ DRIVE SELECTION PANEL ═══════════════════════════════════════╗
        drive_panel = ModernWidget(drive_tab)
        drive_panel.pack(fill=tk.X, pady=(10, 10))

        # Drive frame for selecting drive paths
        drive_frame = ModernWidget(drive_panel)
        drive_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(drive_frame, text="Select Drive:", style='Modern.TLabel').pack(side=tk.LEFT)
        self.drive_entry = ttk.Entry(drive_frame, width=50, style='Modern.TEntry')
        self.drive_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(drive_frame, text="Browse", command=self.browse_drive, style='Modern.TButton').pack(side=tk.LEFT)

        # Drive crawling controls
        crawl_frame = ModernWidget(drive_panel)
        crawl_frame.pack(fill=tk.X, pady=10)
        self.crawl_button = ttk.Button(crawl_frame, text="Crawl Drive", command=self.start_crawl, style='Modern.TButton')
        self.crawl_button.pack(side=tk.LEFT)
        self.stop_button = ttk.Button(crawl_frame, text="Stop Crawl", command=self.stop_crawl, style='Modern.TButton')
        self.stop_button.pack(side=tk.LEFT, padx=(5, 0))
        self.stop_button.config(state=tk.DISABLED)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(crawl_frame, variable=self.progress_var, maximum=100, style='Modern.Horizontal.TProgressbar')
        self.progress_bar.pack(side=tk.LEFT, padx=(10, 0), expand=True, fill=tk.X)
        self.status_label = ttk.Label(crawl_frame, text="Ready", style='Modern.TLabel')
        self.status_label.pack(side=tk.LEFT, padx=(10, 0))

        # Add a text widget for displaying crawl messages
        self.crawl_messages = scrolledtext.ScrolledText(drive_tab, height=10, width=70, bg='#222222', fg='#FFFFFF', font=('Segoe UI', 10))
        self.crawl_messages.pack(pady=(10, 5), fill=tk.BOTH, expand=True)
        self.crawl_messages.config(state=tk.DISABLED)  # Make it read-only

        # ╔═ SEARCH TAB ══════════════════════════════════════════════════╗
        # Search tab to query indexed drives
        search_frame = ModernWidget(notebook)
        notebook.add(search_frame, text='Search')

        # Add databases frame
        db_frame = ModernWidget(search_frame)
        db_frame.pack(fill=tk.X, pady=(10, 5))
        ttk.Label(db_frame, text="Available Databases:", style='Modern.TLabel').pack(side=tk.LEFT)
        self.db_listbox = tk.Listbox(db_frame, bg='#222222', fg='#FFFFFF', font=('Segoe UI', 10), selectmode=tk.MULTIPLE)
        self.db_listbox.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Search query input
        search_input_frame = ModernWidget(search_frame)
        search_input_frame.pack(fill=tk.X, pady=(10, 5))
        ttk.Label(search_input_frame, text="Search Query:", style='Modern.TLabel').pack(side=tk.LEFT)
        self.query_entry = ttk.Entry(search_input_frame, width=50, style='Modern.TEntry')
        self.query_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(search_input_frame, text="Search", command=self.perform_search, style='Modern.TButton').pack(side=tk.LEFT)

        # Search results display
        results_frame = ModernWidget(search_frame)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.results_listbox = tk.Listbox(results_frame, bg='#222222', fg='#FFFFFF', font=('Segoe UI', 10))
        self.results_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_listbox.config(yscrollcommand=scrollbar.set)
        self
        # Button to open selected file
        open_button = ttk.Button(search_frame, text="Open Selected File", command=self.open_selected_file, style='Modern.TButton')
        open_button.pack(pady=(0, 10))

        # ╔═ CHAT TAB ════════════════════════════════════════════════════╗
        # Chat tab for interacting with the AI
        chat_frame = ModernWidget(notebook)
        notebook.add(chat_frame, text='Chat')

        # Styled ScrolledText for conversation
        self.conversation_text = scrolledtext.ScrolledText(chat_frame, height=15, width=70, bg='#222222', fg='#FFFFFF', font=('Segoe UI', 10), wrap=tk.WORD, borderwidth=0, highlightthickness=0)
        self.conversation_text.pack(pady=(10, 5), fill=tk.BOTH, expand=True)

        # Custom scrollbar style for conversation
        conversation_scrollbar = ttk.Scrollbar(chat_frame, command=self.conversation_text.yview, style='Modern.Vertical.TScrollbar')
        conversation_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.conversation_text.config(yscrollcommand=conversation_scrollbar.set)

        # Input frame for user input and buttons
        input_frame = ModernWidget(chat_frame)
        input_frame.pack(fill=tk.X, pady=(5, 10))
        
        # User input box with placeholder
        self.user_input = tk.Text(input_frame, height=3, bg='#222222', fg='#FFFFFF', font=('Segoe UI', 10), wrap=tk.WORD, borderwidth=0, highlightthickness=0)
        self.user_input.pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
        self.user_input.insert("1.0", "Type your message here...")
        self.user_input.bind("<FocusIn>", self.clear_placeholder)
        self.user_input.bind("<FocusOut>", self.add_placeholder)
        
        # Styled Send button
        send_button = ttk.Button(input_frame, text="Send", command=self.send_message, style='Modern.TButton')
        send_button.pack(side=tk.LEFT, padx=(5, 0))

        # Styled Clear button
        clear_button = ttk.Button(input_frame, text="Clear", command=self.clear_conversation, style='Modern.TButton')
        clear_button.pack(side=tk.LEFT, padx=(5, 0))

    # ╔═════════════════════════════════════════════════════════════════╗
    # ║ LOAD WORM FRAMES                                                ║
    # ╠═════════════════════════════════════════════════════════════════╣
    # ║ Loads frames for the worm animation used in the loading screen. ║
    # ╚════════════════════════════════════════════════════════════════╝
    def load_worm_frames(self):
        try:
            self.worm_frames = []
            for i in range(1, 6):  # Assuming you have 5 frames: worm_frame1.png to worm_frame5.png
                image_path = os.path.join(os.path.dirname(__file__), "images", f"worm_frame{i}.png")
                print(f"Loading worm frame: {image_path}")
                image = Image.open(image_path).convert("RGBA")  # Ensure images have transparency
                image = image.resize((80, 80), Image.LANCZOS)
                self.worm_frames.append(ImageTk.PhotoImage(image))
            self.current_frame = 0
            print("All worm frames loaded successfully.")
        except FileNotFoundError as e:
            messagebox.showerror("Error", f"Required worm frame image not found: {e.filename}")
            self.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while loading worm frames: {e}")
            print(f"An error occurred while loading worm frames: {e}")
            import traceback
            traceback.print_exc()
            self.destroy()

    # ╔═════════════════════════════════════════════════════════════════╗
    # ║ BROWSE DRIVE                                                    ║
    # ╠═════════════════════════════════════════════════════════════════╣
    # ║ Opens a file dialog for users to select the drive they want to  ║
    # ║ crawl and updates the entry field with the chosen path.         ║
    # ╚═════════════════════════════════════════════════════════════════╝
    def browse_drive(self):
        drive_path = filedialog.askdirectory()
        if drive_path:
            self.drive_entry.delete(0, tk.END)
            self.drive_entry.insert(0, drive_path)
        else:
            messagebox.showinfo("Hint", "Please select a drive to crawl.")

    # ╔═════════════════════════════════════════════════════════════════╗
    # ║ START CRAWL                                                     ║
    # ╠═════════════════════════════════════════════════════════════════╣
    # ║ Initiates the drive crawling process, taking care to prompt the ║
    # ║ user if a crawl already exists. Starts the crawling in a new    ║
    # ║ thread to keep the UI responsive.                               ║
    # ╚═════════════════════════════════════════════════════════════════╝

    def start_crawl(self):
        print("start_crawl method called")  # Debug print
        drive_path = self.drive_entry.get()
        if not drive_path:
            messagebox.showinfo("Hint", "Please enter a drive path or use the 'Browse' button to select a drive.")
            return

        print(f"Drive path: {drive_path}")  # Debug print

        # Reset and use the existing stop_event
        self.stop_event.clear()
        print("Creating DriveCrawler instance")  # Debug print
        
        # Use a lambda function to call update_crawl_message
        self.crawler = DriveCrawler(
            drive_path, 
            stop_event=self.stop_event, 
            message_callback=lambda msg: self.after(0, lambda: self.update_crawl_message(msg))
        )

        if os.path.exists(self.crawler.index_file):
            update = messagebox.askyesno("Update Index", f"An index for this drive already exists ({self.crawler.index_file}). Do you want to update it?")
            if not update:
                return

        self.crawl_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.progress_var.set(0)  # Reset progress bar to 0
        self.status_label.config(text="Crawling...")
        self.animate_worm()  # Start worm animation when crawling begins

        self.update_crawl_message("Starting crawl...")

        print("Starting crawl thread")  # Debug print
        thread = threading.Thread(target=self.crawl_drive, args=(drive_path,), daemon=True)
        thread.start()

    def update_crawl_message(self, message):
        print(f"Updating crawl message: {message}")  # Debug print
        self.crawl_messages.config(state=tk.NORMAL)
        self.crawl_messages.insert(tk.END, message + "\n")
        self.crawl_messages.see(tk.END)
        self.crawl_messages.config(state=tk.DISABLED)
        self.update_idletasks()  # Force GUI update

    # ╔═════════════════════════════════════════════════════════════════╗
    # ║ STOP CRAWL                                                      ║
    # ╠═════════════════════════════════════════════════════════════════╣
    # ║ Sets the stop_event flag to signal the ongoing crawling thread  ║
    # ║ to terminate gracefully. Also manages button states and stops   ║
    # ║ animations.                                                     ║
    # ╚═════════════════════════════════════════════════════════════════╝

    def stop_crawl(self):
        self.stop_event.set()  # Signal the crawling thread to stop
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Stopping crawl... Please wait.")
        self.stop_worm()

    # ╔═════════════════════════════════════════════════════════════════╗
    # ║ CRAWL DRIVE                                                     ║
    # ╠═════════════════════════════════════════════════════════════════╣
    # ║ Performs the actual drive crawling by calling DriveCrawler's    ║
    # ║ crawl_drive method. A callback function is used to update the   ║
    # ║ UI with crawl progress.                                         ║
    # ╚═════════════════════════════════════════════════════════════════╝

    def crawl_drive(self, drive_path):
        def update_progress(current, total):
            progress = (current / total) * 100
            print(f"Updating progress: {progress:.2f}%")
            self.progress_var.set(progress)
            self.update_idletasks()
            self.update_crawl_message(f"Processed {current} of {total} files ({progress:.2f}%)")

        print("Crawl drive method called")
        self.update_crawl_message("Counting total number of supported files...")
        self.crawler.crawl_drive(callback=update_progress)
        print("Crawl completed, calling crawl_complete")
        self.after(0, self.crawl_complete)

    # ╔═════════════════════════════════════════════════════════════════╗
    # ║ CRAWL COMPLETE                                                  ║
    # ╠═════════════════════════════════════════════════════════════════╣
    # ║ Handles the tasks to perform once the crawling operation is     ║
    # ║ completed, such as updating buttons, stopping animations, and   ║
    # ║ informing the user via messagebox.                              ║
    # ╚═════════════════════════════════════════════════════════════════╝

    def crawl_complete(self):
        print("Crawl complete method called")  # Debug print
        self.crawl_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.stop_worm()
        self.load_databases()  # Refresh the database list
        
        if hasattr(self, 'crawler') and hasattr(self.crawler, 'index_file'):
            messagebox.showinfo("Crawl Complete", f"Drive crawl has finished. Index saved as {self.crawler.index_file}")
            
            # Ensure the newly created index is selected in the listbox
            index_name = os.path.basename(self.crawler.index_file)
            items = self.db_listbox.get(0, tk.END)
            if index_name in items:
                index_position = items.index(index_name)
                self.db_listbox.selection_set(index_position)
                self.db_listbox.see(index_position)
            else:
                print(f"Warning: New index {index_name} not found in the listbox. Refreshing database list.")
                self.load_databases()  # Try to load databases again
                
                # Check again after refreshing
                items = self.db_listbox.get(0, tk.END)
                if index_name in items:
                    index_position = items.index(index_name)
                    self.db_listbox.selection_set(index_position)
                    self.db_listbox.see(index_position)
                else:
                    print(f"Error: New index {index_name} still not found after refresh.")
        else:
            messagebox.showinfo("Crawl Complete", "Drive crawl has finished, but no index was created.")

        # Force update of the GUI
        self.update_idletasks()

    # ╔═════════════════════════════════════════════════════════════════╗
    # ║ PERFORM SEARCH                                                  ║
    # ╠═════════════════════════════════════════════════════════════════╣
    # ║ Initiates a search across the selected databases. If a query is ║
    # ║ provided, it launches the search in a separate thread to avoid  ║
    # ║ blocking the main UI.                                           ║
    # ╚═════════════════════════════════════════════════════════════════╝

    def perform_search(self):
        try:
            print("Starting perform_search")
            query = self.query_entry.get()
            if not query:
                messagebox.showinfo("Hint", "Please enter a search query before clicking 'Search'.")
                return

            selected_dbs = [self.db_listbox.get(i) for i in self.db_listbox.curselection()]
            if not selected_dbs:
                messagebox.showwarning("Warning", "Please select at least one database.")
                return

            self.status_label.config(text="Searching...")
            self.progress_var.set(0)
            self.animate_worm()

            thread = threading.Thread(target=self.search_thread, args=(query, selected_dbs), daemon=True)
            thread.start()
            print("Started search_thread")
        except Exception as e:
            print(f"An error occurred in perform_search: {e}")
            import traceback
            traceback.print_exc()

    # ╔═════════════════════════════════════════════════════════════════╗
    # ║ SEARCH THREAD                                                   ║
    # ╠═════════════════════════════════════════════════════════════════╣
    # ║ Runs the search operation for the provided query across the     ║
    # ║ selected databases. Filters the results and sorts them by       ║
    # ║ relevance, then displays them in the UI.                        ║
    # ╚═════════════════════════════════════════════════════════════════╝

    def search_thread(self, query, selected_dbs):
        try:
            print("Executing search_thread")
            all_results = []
            query = query.strip().lower()  # Trim and convert to lowercase
            for db in selected_dbs:
                crawler = self.crawlers.get(db)
                if crawler:
                    results, _ = crawler.search(query)
                    all_results.extend(results)
                else:
                    print(f"No crawler found for database: {db}")
            
            # More flexible filtering
            filtered_results = []
            for res in all_results:
                file_name = res['file_name'].lower()
                if any(part in file_name for part in query.split()):
                    filtered_results.append(res)
            
            print(f"Filtered down to {len(filtered_results)} results related to '{query}'.")
            
            # Remove duplicates
            unique_results = {res['file_path']: res for res in filtered_results}.values()
            
            # Sort by distance (ascending order)
            sorted_results = sorted(unique_results, key=lambda x: x['distance'])
            
            self.after(0, self.display_search_results, sorted_results)
            print("Completed search_thread with filtered results")
        except Exception as e:
            print(f"An error occurred in search_thread: {e}")
            import traceback
            traceback.print_exc()

    # ╔═════════════════════════════════════════════════════════════════╗
    # ║ DISPLAY SEARCH RESULTS                                          ║
    # ╠═════════════════════════════════════════════════════════════════╣
    # ║ Displays the search results in the listbox widget, sorted by    ║
    # ║ relevance. Stops the worm animation and updates the UI status.  ║
    # ╚═════════════════════════════════════════════════════════════════╝

    def display_search_results(self, results):
        try:
            print("Displaying search results")
            self.stop_worm()
            self.status_label.config(text="Search complete")
            self.progress_var.set(100)

            self.results_listbox.delete(0, tk.END)
            for result in results:
                # Use the full file name including extension
                full_file_name = result['file_name']
                # Format the display text with full file name and relevance score
                display_text = f"{full_file_name} - Relevance: {1 / (1 + result['distance']):.2f}"
                self.results_listbox.insert(tk.END, display_text)

            if not results:
                messagebox.showinfo("Search Results", "No results found. Try a different search query or crawl more drives.")
            else:
                with self.search_results_lock:
                    self.search_results = results  # Update within a lock to ensure thread safety
                messagebox.showinfo("Hint", "Double-click on a search result to open the file.")
                print(f"Displayed {len(results)} search results.")
        except Exception as e:
            print(f"An error occurred in display_search_results: {e}")
            import traceback
            traceback.print_exc()

    # ╔═════════════════════════════════════════════════════════════════╗
    # ║ ANIMATE WORM                                                    ║
    # ╠═════════════════════════════════════════════════════════════════╣
    # ║ Animates the worm graphic in the canvas to visually indicate    ║
    # ║ ongoing activity. Restarts from the beginning once the worm     ║
    # ║ moves beyond the canvas boundaries.                             ║
    # ╚════════════════════════════════════════════════════════════════╝

    def animate_worm(self):
        if self.worm is None:
            self.worm = self.canvas.create_image(50, 20, image=self.worm_frames[self.current_frame], anchor='nw')  # Adjusted y-coordinate to 20
        
        self.current_frame = (self.current_frame + 1) % len(self.worm_frames)
        self.canvas.itemconfig(self.worm, image=self.worm_frames[self.current_frame])
        self.canvas.move(self.worm, 5, 0)

        worm_pos = self.canvas.coords(self.worm)
        canvas_width = self.canvas.winfo_width()
        if worm_pos[0] > canvas_width:
            self.canvas.coords(self.worm, -80, 20)  # Reset position with y=20
        
        self.worm_animation = self.after(150, self.animate_worm)  # Adjusted speed if needed

    # ╔═══════════════════════════════════════════════════════════════╗
    # ║ STOP WORM                                                       ║
    # ╠════════════════════════════════════════════════════════════════╣
    # ║ Stops the worm animation by canceling the scheduled animation   ║
    # ║ callback and deleting the worm from the canvas.                 ║
    # ╚═══════════════════════════════════════════════════════════════╝

    def stop_worm(self):
        if hasattr(self, 'worm_animation'):
            self.after_cancel(self.worm_animation)
        if self.worm is not None:
            self.canvas.delete(self.worm)
            self.worm = None

    # ╔═════════════════════════════════════════════════════════════════╗
    # ║ SEND MESSAGE                                                    ║
    # ╠═════════════════════════════════════════════════════════════════╣
    # ║ Retrieves the user's input message, adds it to the conversation ║
    # ║ text, and then starts a thread to get a response from the AI.   ║
    # ╚════════════════════════════════════════════════════════════════╝

    def send_message(self):
            user_message = self.user_input.get("1.0", tk.END).strip()
            if not user_message or user_message == "Type your message here...":
                messagebox.showinfo("Hint", "Please enter a message before sending.")
                return

            self.conversation_text.config(state=tk.NORMAL)  # Enable editing
            self.conversation_text.insert(tk.END, f"You: {user_message}\n\n")
            self.conversation_text.config(state=tk.DISABLED)  # Disable editing
            self.conversation_text.see(tk.END)  # Scroll to the end of the conversation

            self.user_input.delete("1.0", tk.END)
            self.add_placeholder(None)  # Re-add the placeholder text

            # Use threading to prevent GUI freezing and set as daemon
            threading.Thread(target=self.get_ai_response, args=(user_message,), daemon=True).start()

    def get_relevant_contents_for_chat(self):
            """
            Retrieves context from selected files to provide to the AI.
            Since we only have file names, we'll use them as context.
            """
            relevant_contents = ""
            selected_indices = self.results_listbox.curselection()
            if not selected_indices:
                return relevant_contents

            for index in selected_indices:
                result = self.search_results[index]
                file_name = os.path.splitext(result['file_name'])[0].replace('_', ' ').title()
                file_path = result['file_path']
                relevant_contents += f"File: {file_name}\nPath: {file_path}\n\n"

            return relevant_contents
    # ╔═════════════════════════════════════════════════════════════════╗
    # ║ GENERATE CACHED RESPONSE                                        ║
    # ╠═════════════════════════════════════════════════════════════════╣
    # ║ Generates an AI response using a cached model response. This    ║
    # ║ function is cached to improve efficiency on repeated prompts.   ║
    # ╚═════════════════════════════════════════════════════════════════╝

    @lru_cache(maxsize=100)
    def generate_cached_response(self, prompt):
        if self.model is None or self.tokenizer is None:
            return "I'm sorry, but the AI model is not available at the moment. Please try again later."
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=128000)
        attention_mask = (inputs != self.tokenizer.pad_token_id).long()
        inputs = inputs.to(self.device)
        attention_mask = attention_mask.to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                attention_mask=attention_mask,
                max_length=512,  # Limit output length
                num_return_sequences=1,
                temperature=0.7,  # Adjust temperature for faster generation
                top_k=50,
                top_p=0.95,
                do_sample=True,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ╔═════════════════════════════════════════════════════════════════╗
    # ║ GET AI RESPONSE                                                 ║
    # ╠════════════════════���═════���══════════════════════════════════════╣
    # ║ Handles generating the AI response based on user input. Uses a  ║
    # ║ thread pool to ensure timeout control on response generation.   ║
    # ╚═════════════════════════════════════════════════════════════════╝

    def get_ai_response(self, user_message):
            try:
                print("Starting get_ai_response")
                if self.model is None or self.tokenizer is None:
                    print("Error: Model or tokenizer is not initialized")
                    return self.update_conversation("I'm sorry, the AI model is not ready. Please try again later.")
                
                self.after(0, self.start_chat_worm_animation)  # Start worm animation

                # Retrieve context from selected files
                relevant_contents = self.get_relevant_contents_for_chat()
                prompt = self.create_prompt(user_message, relevant_contents)
                
                print(f"Generated prompt: {prompt[:500]}...")  # Print first 500 characters of prompt
                
                try:
                    print("Generating AI response...")
                    start_time = time.time()
                    
                    def generate_response():
                        try:
                            response = self.generate_cached_response(prompt)
                            generation_time = time.time() - start_time
                            print(f"Response generated in {generation_time:.2f} seconds")
                            print(f"Response preview: {response[:100]}...")  # Print first 100 chars
                            return response
                        except Exception as e:
                            print(f"Error in generate_response: {e}")
                            traceback.print_exc()
                            return None

                    # Use ThreadPoolExecutor to run the generation with a timeout
                    with ThreadPoolExecutor() as executor:
                        future = executor.submit(generate_response)
                        try:
                            ai_response = future.result(timeout=60)  # Increased timeout to 60 seconds
                        except TimeoutError:
                            print("Main model timed out, attempting fallback response")
                            ai_response = self.generate_fallback_response(prompt)

                    if ai_response:
                        # Schedule the GUI update on the main thread
                        self.after(0, lambda: self.update_conversation(ai_response))
                    else:
                        print("AI response is empty, displaying error message")
                        self.after(0, lambda: self.update_conversation("I'm sorry, I couldn't generate a response. Please try again."))
    
                except Exception as e:
                    error_message = f"An error occurred while generating AI response: {e}"
                    print(error_message)
                    traceback.print_exc()
                    self.after(0, lambda: self.update_conversation("I'm sorry, I couldn't process your request at the moment."))
            except Exception as e:
                error_message = f"An error occurred in get_ai_response: {e}"
                print(error_message)
                traceback.print_exc()
                self.after(0, lambda: self.update_conversation("I apologize, but an error occurred while processing your request."))

    def generate_fallback_response(self, prompt):
        # Implement a simpler, faster response generation method here
        # This could be a rule-based system or a smaller, faster model
        return "I apologize, but I'm having trouble generating a detailed response at the moment. Could you please rephrase your question or try again later?"

    # ╔═════════════════════════════════════════════════════════════════╗
    # ║ GET RELEVANT CONTENTS                                           ║
    # ╠═════════════════════════════════════════════════════════════════╣
    # ║ Retrieves the relevant contents from the search results to      ║
    # ║ provide context to the AI prompt. Limits the number of results  ║
    # ║ for brevity and efficiency.                                     ║
    # ╚═════════════════════════════════════════════════════════════════╝

    def get_relevant_contents(self):
            relevant_contents = ""
            with self.search_results_lock:
                current_search_results = self.search_results.copy()
            
            if current_search_results:
                for result in current_search_results:
                    file_path = result['file_path']
                    if os.path.exists(file_path):
                        try:
                            with open(file_path, 'r', encoding='utf-8') as file:
                                content = file.read()
                                relevant_contents += f"File: {os.path.basename(file_path)}\n"
                                relevant_contents += f"Content: {content[:500]}...\n\n"  # First 500 characters
                        except Exception as e:
                            print(f"Error reading file {file_path}: {e}")
        
            return relevant_contents

    # ╔═════════════════════════════════════════════════════════════════╗
    # ║ CREATE PROMPT                                                   ║
    # ╠═════════════════════════════════════════════════════════════════╣
    # ║ Creates the input prompt for the AI based on user message and   ║
    # ║ any relevant contents from previous search results.             ║
    # ╚═════════════════════════════════════════════════════════════════╝

    def create_prompt(self, user_message, relevant_contents):
        if relevant_contents:
            return f"Context:\n{relevant_contents}\nBased on the above context, please answer the following question:\nHuman: {user_message}\nAI:"
        else:
            return f"Human: {user_message}\nAI:"

    # ╔═════════════════════════════════════════════════════════════════╗
    # ║ UPDATE CONVERSATION                                             ║
    # ╠═════════════════════════════════════════════════════════════════╣
    # ║ Updates the conversation text box with the AI's response,       ║
    # ║ ensuring the response is displayed alongside the chat logo.     ║
    # ║ Stops the worm animation after updating.                        ║
    # ╚═════════════════════════════════════════════════════════════════╝

    def update_conversation(self, ai_response):
        try:
            print(f"Updating conversation with AI response: '{ai_response}'")
            self.conversation_text.config(state=tk.NORMAL)  # Enable editing
            
            # Insert the logo
            self.conversation_text.image_create(tk.END, image=self.chat_logo_photo)
            self.conversation_text.insert(tk.END, " ")  # Add a space after the logo
            
            # Insert the response text
            if ai_response.strip():
                self.conversation_text.insert(tk.END, f"{ai_response}\n\n")
            else:
                self.conversation_text.insert(tk.END, "I'm sorry, I couldn't generate a response.\n\n")
            
            self.conversation_text.config(state=tk.DISABLED)  # Disable editing
            self.conversation_text.see(tk.END)  # Scroll to the end of the conversation
            print("AI response displayed in conversation")  # Fixed this line
            
            # Force update the GUI
            self.update_idletasks()
        except Exception as e:
            error_message = f"An error occurred in update_conversation: {e}"
            print(error_message)
            traceback.print_exc()
            messagebox.showerror("Update Conversation Error", error_message)
        finally:
            self.stop_chat_worm_animation()  # Ensure animation stops even if there's an error

    # ╔═════════════════════════════════════════════════════════════════╗
    # ║ CLEAR CONVERSATION                                              ║
    # ╠═════════════════════════════════════════════════════════════════╣
    # ║ Clears the conversation area in the GUI and optionally calls    ║
    # ║ the clear_conversation method from the DriveCrawler.            ║
    # ╚═════════════════════════════════════════════════════════════════╝
    
    def clear_conversation(self):
        """
        Clears the conversation text area in the GUI.
        Optionally, clears any conversation history in the crawler.
        """
        self.conversation_text.delete("1.0", tk.END)
        if self.crawler:
            self.crawler.clear_conversation()  # Ensure this method exists in drive_crawler.py

    # ╔═════════════════════════════════════════════════════════════════╗
    # ║ OPEN SELECTED FILE                                              ║
    # ╠═════════════════════════════════════════════════════════════════╣
    # ║ Attempts to open the selected file from the search results in   ║
    # ║ the default viewer. Works across multiple platforms with proper ║
    # ║ command usage.                                                  ║
    # ╚═════════════════════════════════════════════════════════════════╝

    def open_selected_file(self, event=None):
        selection = self.results_listbox.curselection()
        if selection:
            index = selection[0]
            file_path = self.search_results[index]['file_path']
            try:
                os.startfile(file_path)  # For Windows
            except AttributeError:
                # os.startfile is not available on Mac or Linux
                try:
                    if os.name == 'posix':
                        if sys.platform == 'darwin':
                            subprocess.call(['open', file_path])  # macOS
                        else:
                            subprocess.call(['xdg-open', file_path])  # Linux
                except Exception as e:
                    messagebox.showerror("Error", f"Unable to open file: {e}")
            except Exception as e:
                messagebox.showerror("Error", f"Unable to open file: {e}")
        else:
            messagebox.showinfo("Info", "Please select a file to open.")

    # ╔═════════════════════════════════════════════════════════════════╗
    # ║ LOAD DATABASES                                                  ║
    # ╠═════════════════════════════════════════════════════════════════╣
    # ║ Loads all existing database indexes into the listbox for easy   ║
    # ║ selection during searches. Initializes DriveCrawler instances   ║
    # ║ for each available index.                                       ║
    # ╚═════════════════════════════════════════════════════════════════╝

    def load_databases(self):
        self.db_listbox.delete(0, tk.END)  # Clear existing items
        print("Scanning for index files...")
        
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"Script directory: {script_dir}")
        
        # Search for index files in the script directory and current working directory
        index_files = []
        for search_dir in [script_dir, os.getcwd()]:
            index_files.extend([
                file for file in os.listdir(search_dir) 
                if file.startswith('index_') and file.endswith('.pkl')
            ])
        
        if index_files:
            for file in index_files:
                print(f"Found index file: {file}")
                self.db_listbox.insert(tk.END, file)
                # Initialize a DriveCrawler for each index
                if file not in self.crawlers:
                    crawler = DriveCrawler(
                        root_path=self.drive_entry.get(),  # Ensure root_path is correctly set
                        stop_event=self.stop_event
                    )
                    crawler.index_file = os.path.join(search_dir, file)  # Use full path
                    crawler.load_index()
                    self.crawlers[file] = crawler
        else:
            print("No index files found.")
            self.db_listbox.insert(tk.END, "No databases found")
        
        print(f"Loaded {self.db_listbox.size()} databases")
        
        # Force update of the listbox
        self.db_listbox.update_idletasks()

        # Debug: Print current working directory and list all files
        print(f"Current working directory: {os.getcwd()}")
        print("Files in current directory:")
        for file in os.listdir(os.getcwd()):
            print(f"  {file}")

    # ╔═════════════════════════════════════════════════════════════════╗
    # ║ DEICONIFY WINDOW                                                ║
    # ╠═════════════════════════════════════════════════════════════════╣
    # ║ Restores the window from a minimized state and brings it to the ║
    # ║ front of all other windows.                                     ║
    # ╚═════════════════════════════════════════════════════════════════╝
    
    def deiconify_window(self, event=None):
        self.deiconify()  # Restores the window if minimized.
        self.lift()  # Bring the window to the front.

    # ╔════════════════════════════════════════════════════���════════════╗
    # ║ ON CLOSING                                                      ║
    # ╠═════════════════════════════════════════════════════════════════╣
    # ║ Handles the event when the application window is closed,        ║
    # ║ ensuring proper resource cleanup.                               ║
    # ╚═════════════════════════════════════════════════════════════════╝
    def on_closing(self):
        print("Cleaning up and closing application...")
        # Perform any necessary cleanup here
        self.destroy()

    # ╔═════════════════════════════════════════════════════════════════╗
    # ║ TOGGLE ALWAYS ON TOP                                            ║
    # ╠═════════════════════════════════════════════════════════════════╣
    # ║ Toggles the 'always on top' state of the application window,    ║
    # ║ and updates the button to reflect the change.                   ║
    # ╚═════════════════════════════════════════════════════════════════╝
    
    def toggle_always_on_top(self):
        if self.attributes('-topmost'):
            # If the window is already set to always be on top, disable it.
            self.attributes('-topmost', False)
            self.always_on_top_button.config(text="📌", bg='#333333')  # Update button style.
        else:
            # Set the window to always be on top.
            self.attributes('-topmost', True)
            self.always_on_top_button.config(text="📌", bg='#555555')  # Update button style.

    # ╔═════════════════════════════════════════════════════════════════╗
    # ║ START CHAT WORM ANIMATION                                       ║
    # ╠═════════════════════════════════════════════════════════════════╣
    # ║ Starts the worm animation used as a loading indicator while the ║
    # ║ AI response is being generated.                                 ║
    # ╚═════════════════════════════════════════════════════════════════╝
    
    def start_chat_worm_animation(self):
        # Create a new canvas for the chat worm animation if it doesn't exist
        if not hasattr(self, 'chat_worm_canvas'):
            self.chat_worm_canvas = tk.Canvas(self.conversation_text, width=400, height=80, bg='#333333', highlightthickness=0)
            self.chat_worm_canvas.pack(side=tk.BOTTOM, pady=10)
        
        # Add a loading text to indicate the AI is generating a response.
        self.loading_text = self.chat_worm_canvas.create_text(200, 60, text="Generating response...", fill="#FFFFFF", font=('Segoe UI', 10))
        
        # Start the worm animation.
        self.animate_chat_worm()

    # ╔═════════════════════════════════════════════════════════════════╗
    # ║ ANIMATE CHAT WORM                                               ║
    # ╠═════════════════════════════════════════════════════════════════╣
    # ║ Handles the animation of the worm image during the chat loading ║
    # ║ phase, providing visual feedback to the user.                   ║
    # ╚═════════════════════════════════════════════════════════════════╝
    
    def animate_chat_worm(self):
        if not hasattr(self, 'chat_worm'):
            # Create the first worm image on the canvas.
            self.chat_worm = self.chat_worm_canvas.create_image(0, 20, image=self.worm_frames[0], anchor='nw')
        
        # Move to the next frame of the worm animation.
        self.current_chat_frame = (getattr(self, 'current_chat_frame', 0) + 1) % len(self.worm_frames)
        self.chat_worm_canvas.itemconfig(self.chat_worm, image=self.worm_frames[self.current_chat_frame])
        self.chat_worm_canvas.move(self.chat_worm, 5, 0)

        worm_pos = self.chat_worm_canvas.coords(self.chat_worm)  # Get the worm's current position.
        canvas_width = self.chat_worm_canvas.winfo_width()  # Get the canvas width.

        # If the worm moves beyond the canvas, reset its position.
        if worm_pos[0] > canvas_width:
            self.chat_worm_canvas.coords(self.chat_worm, -80, 20)  # Reset the position.

        # Schedule the next frame of the animation.
        self.chat_worm_animation = self.after(150, self.animate_chat_worm)

    # ╔═════════════════════════════════════════════════════════════════╗
    # ║ STOP CHAT WORM ANIMATION                                        ║
    # ╠═════════════════════════════════════════════════════════════════╣
    # ║ Stops the chat worm loading animation, removes the worm from    ║
    # ║ the canvas, and deletes any associated resources.               ║
    # ╚═════════════════════════════════════════════════════════════════╝
    
    def stop_chat_worm_animation(self):
        # Cancel the scheduled worm animation if it exists.
        if hasattr(self, 'chat_worm_animation'):
            self.after_cancel(self.chat_worm_animation)
        
        # Remove the chat worm canvas if it exists.
        if hasattr(self, 'chat_worm_canvas'):
            self.chat_worm_canvas.pack_forget()
            del self.chat_worm_canvas
        
        # Remove the worm animation instance.
        if hasattr(self, 'chat_worm'):
            del self.chat_worm

        # Remove the loading text instance.
        if hasattr(self, 'loading_text'):
            del self.loading_text

    # ╔═════════════════════════════════════════════════════════════════╗
    # ║ CLEAR PLACEHOLDER                                               ║
    # ╠═════════════════════════════════════════════════════════════════╣
    # ║ Clears the placeholder text in the user input box if it is      ║
    # ║ currently displaying the default placeholder.                   ║
    # ╚═════════════════════════════════════════════════════════════════╝
    
    def clear_placeholder(self, event):
        if self.user_input.get("1.0", "end-1c") == "Type your message here...":
            # Clear the text box and change text color to indicate user input.
            self.user_input.delete("1.0", tk.END)
            self.user_input.config(fg='#FFFFFF')

    # ╔═════════════════════════════════════════════════════════════════╗
    # ║ ADD PLACEHOLDER                                                 ║
    # ╠═════════════════════════════════════════════════════════════════╣
    # ║ Adds the placeholder text back to the user input box if it is   ║
    # ║ currently empty, changing the text color to indicate it.        ║
    # ╚═════════════════════════════════════════════════════════════════╝
    
    def add_placeholder(self, event):
        if not self.user_input.get("1.0", "end-1c"):
            # Add placeholder text and change color to indicate it is a hint.
            self.user_input.insert("1.0", "Type your message here...")
            self.user_input.config(fg='#888888')

    def create_tooltip(self, widget, text):
        tooltip = tk.Label(self, text=text, background="#ffffe0", relief="solid", borderwidth=1)
        tooltip.pack_forget()

        def enter(event):
            tooltip.lift(widget)
            tooltip.place(in_=widget, x=0, y=widget.winfo_height())

        def leave(event):
            tooltip.place_forget()

        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)

    def show_help(self):
        help_text = """
        How to use DriveWorm:
        1. Enter a drive path or click 'Browse' to select a drive.
        2. Click 'Crawl' to index the files on the drive.
        3. Enter a search query and click 'Search' to find files.
        4. Double-click on a search result to open the file.
        5. Type a message in the chat box and click 'Send' to interact with the AI.
        6. The AI will provide responses based on the indexed files and your queries.
        """
        messagebox.showinfo("DriveWorm Help", help_text)

    # ╔═════════════════════════════════════════════════════════════════╗
    # ║ MAIN APPLICATION ENTRY POINT                                    ║
    # ╠═════════════════════════════════════════════════════════════════╣
    # ║ The entry point for running the DriveWorm application. If an    ║
    # ║ error occurs during execution, it will be printed to the console║
    # ║ and the stack trace will be provided.                           ║
    # ╚═════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    try:
        app = DriveWormApp()
        app.after(100, lambda: app.protocol("WM_DELETE_WINDOW", app.on_closing))
        app.mainloop()
    except Exception as e:
        print(f"An error occurred during application startup: {e}")
        import traceback
        traceback.print_exc()
