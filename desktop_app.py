import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import os

# --- Configuration ---
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Define paths and class names for *both* models
# Ensure these file names exactly match your .h5 files in 'trained_medical_models'
MODELS_CONFIG = {
    "covid": {
        "path": 'trained_medical_models/COVID19_XRay_Classifier.h5', #
        "class_names": ['COVID', 'Normal', 'Pneumonia']
    },
    "brain_tumor": {
        "path": 'trained_medical_models/Brain_Tumor_MRI_Classifier.h5', #
        "class_names": ['glioma', 'notumor', 'meningioma', 'pituitary']
    }
}

# --- Load both models once when the application starts ---
loaded_models = {}
for model_type, config in MODELS_CONFIG.items():
    model_path = config["path"]
    try:
        model = tf.keras.models.load_model(model_path)
        loaded_models[model_type] = model
        print(f"Model '{model_path}' ({model_type}) loaded successfully.")
    except Exception as e:
        messagebox.showwarning("Model Load Error", f"Error loading model '{model_path}' ({model_type}):\n{e}\nThis model type will be unavailable.")
        print(f"Error loading model '{model_path}' ({model_type}): {e}")
        loaded_models[model_type] = None # Set model to None if loading fails

# --- Preprocessing function for a single image ---
def preprocess_image(img_path):
    """
    Loads an image, resizes it, converts to RGB, normalizes pixel values,
    and adds a batch dimension, ready for model prediction.
    """
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((IMG_HEIGHT, IMG_WIDTH))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array
    except Exception as e:
        messagebox.showerror("Image Processing Error", f"Failed to process image: {e}")
        return None

# --- GUI Application Logic ---
class MedicalDiagnosisApp:
    def __init__(self, master):
        self.master = master
        master.title("Medical Image Diagnosis") # Updated title
        master.geometry("800x750") # Slightly increased height for better spacing
        master.resizable(False, False)

        # Styling
        master.configure(bg="#e6f7ff") # Light blue background
        font_large = ("Arial", 16, "bold")
        font_medium = ("Arial", 12)

        # Title Label
        self.title_label = tk.Label(master, text="Diagnostic Vision and Smartscan Med", font=("Arial", 20, "bold"), bg="#e6f7ff", fg="#0056b3")
        self.title_label.pack(pady=15)

        # Instruction Label
        self.instruction_label = tk.Label(master, text="Select a model and upload an image for diagnosis.", font=font_medium, bg="#e6f7ff", fg="#333")
        self.instruction_label.pack(pady=5)

        # --- Model Selection Dropdown ---
        self.model_type_frame = tk.Frame(master, bg="#e6f7ff")
        self.model_type_frame.pack(pady=10)

        tk.Label(self.model_type_frame, text="Choose Diagnosis Type:", font=font_medium, bg="#e6f7ff", fg="#0056b3").pack(side=tk.LEFT, padx=10)

        # Get available model types (only those that loaded successfully)
        available_model_types = [
            "COVID-19 X-ray" if loaded_models.get("covid") else None,
            "Brain Tumor MRI" if loaded_models.get("brain_tumor") else None
        ]
        available_model_types = [m for m in available_model_types if m is not None]

        if not available_model_types:
            messagebox.showerror("No Models Available", "No machine learning models could be loaded. Application cannot function.")
            master.destroy() # Close window if no models are available
            return

        self.selected_model_type = tk.StringVar(master)
        self.selected_model_type.set(available_model_types[0]) # Default to the first available model

        # Create a mapping from display name to internal key
        self.display_to_internal_model_map = {
            "COVID-19 X-ray": "covid",
            "Brain Tumor MRI": "brain_tumor"
        }

        # Create the OptionMenu (dropdown)
        self.model_option_menu = tk.OptionMenu(
            self.model_type_frame,
            self.selected_model_type,
            *available_model_types # Unpack the list of choices
        )
        self.model_option_menu.config(font=font_medium, bg="white", activebackground="#e0e0e0")
        self.model_option_menu["menu"].config(font=font_medium, bg="white")
        self.model_option_menu.pack(side=tk.LEFT, padx=10)

        # Frame for Browse/Get Diagnosis buttons
        self.button_frame = tk.Frame(master, bg="#e6f7ff")
        self.button_frame.pack(pady=10)

        self.browse_button = tk.Button(self.button_frame, text="Browse...", command=self.load_image, font=font_medium, bg="#28a745", fg="white", relief=tk.RAISED, bd=3)
        self.browse_button.pack(side=tk.LEFT, padx=10)

        # File path entry (disabled, shows selected file name)
        self.file_path_var = tk.StringVar(master, value="No file selected.")
        self.file_path_entry = tk.Entry(self.button_frame, textvariable=self.file_path_var, state='readonly', width=30, font=font_medium)
        self.file_path_entry.pack(side=tk.LEFT, padx=10)

        self.predict_button = tk.Button(self.button_frame, text="Get Diagnosis", command=self.predict_image, font=font_medium, bg="#007bff", fg="white", relief=tk.RAISED, bd=3)
        self.predict_button.pack(side=tk.LEFT, padx=10)
        self.predict_button.config(state=tk.DISABLED) # Disable until image is loaded

        # Image Display Area
        self.image_frame = tk.Frame(master, bg="#ffffff", bd=2, relief=tk.GROOVE)
        self.image_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)
        self.image_frame.pack_propagate(False)

        self.image_label = tk.Label(self.image_frame, bg="#ffffff")
        self.image_label.pack(expand=True)

        # Result Display Area
        self.result_frame = tk.Frame(master, bg="#e9f7ff", bd=2, relief=tk.SOLID)
        self.result_frame.pack(pady=15, padx=20, fill=tk.X)

        self.result_title = tk.Label(self.result_frame, text="Diagnosis Result:", font=font_large, bg="#e9f7ff", fg="#007bff")
        self.result_title.pack(pady=(10, 5))

        self.diagnosis_label = tk.Label(self.result_frame, text="Select a model, upload an image, and click 'Get Diagnosis'.", font=("Arial", 14, "bold"), bg="#e9f7ff", fg="#333")
        self.diagnosis_label.pack(pady=5)

        self.confidence_label = tk.Label(self.result_frame, text="", font=font_medium, bg="#e9f7ff", fg="#555")
        self.confidence_label.pack(pady=(0, 10))

        self.current_image_path = None # To store path of selected image

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image", # Generic title for both types
            filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
        )
        if file_path:
            self.current_image_path = file_path
            self.file_path_var.set(os.path.basename(file_path)) # Show filename in entry
            try:
                img = Image.open(file_path)
                max_display_size = (400, 400)
                img.thumbnail(max_display_size, Image.Resampling.LANCZOS)

                self.photo = ImageTk.PhotoImage(img)
                self.image_label.config(image=self.photo)
                self.image_label.image = self.photo # IMPORTANT: Keep a reference!

                self.image_frame.config(width=img.width, height=img.height)

                self.predict_button.config(state=tk.NORMAL)
                self.diagnosis_label.config(text="Click 'Get Diagnosis' to analyze.")
                self.confidence_label.config(text="")
            except Exception as e:
                messagebox.showerror("Image Error", f"Could not load image: {e}")
                self.current_image_path = None
                self.file_path_var.set("No file selected.")
                self.predict_button.config(state=tk.DISABLED)
        else:
            self.file_path_var.set("No file selected.") # Reset if dialog is cancelled
            self.current_image_path = None
            self.predict_button.config(state=tk.DISABLED)

    def predict_image(self):
        if not self.current_image_path:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        # Get the internally recognized model key (e.g., "covid" or "brain_tumor")
        selected_display_type = self.selected_model_type.get()
        selected_internal_type = self.display_to_internal_model_map.get(selected_display_type)

        model_to_use = loaded_models.get(selected_internal_type)
        class_names_to_use = MODELS_CONFIG.get(selected_internal_type)["class_names"]

        if not model_to_use:
            messagebox.showwarning("Model Not Loaded", f"The '{selected_display_type}' model could not be loaded. Please check console for errors or confirm model file exists.")
            return

        self.diagnosis_label.config(text="Analyzing...", fg="#007bff")
        self.confidence_label.config(text="")
        self.master.update_idletasks() # Update GUI to show "Analyzing..."

        processed_img = preprocess_image(self.current_image_path)
        if processed_img is None:
            self.diagnosis_label.config(text="Prediction failed due to image error.", fg="#dc3545")
            return

        try:
            predictions = model_to_use.predict(processed_img)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            predicted_class_name = class_names_to_use[predicted_class_index]
            confidence = float(np.max(predictions)) * 100

            self.diagnosis_label.config(text=f"Predicted Class: {predicted_class_name}", fg="#0056b3")
            self.confidence_label.config(text=f"Confidence: {confidence:.2f}%", fg="#555")
        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred during prediction: {e}")
            self.diagnosis_label.config(text="Prediction failed.", fg="#dc3545")

# --- Main application execution ---
if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TensorFlow GPU warnings

    root = tk.Tk()
    app = MedicalDiagnosisApp(root)
    root.mainloop()
