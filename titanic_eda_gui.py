#This file will create the GUI using tkinter to interact with the user. It will allow the user to upload a CSV file, start the hyperparameter tuning process, and display results.
# titanic_eda_gui.py
import tkinter as tk
from tkinter import filedialog, messagebox
from titanic_model import load_data, tune_model  # Import the functions from titanic_model.py

class TitanicModelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Titanic Survival Prediction & Hyperparameter Tuning")
        
        self.filepath = None  # To store the selected file path
        
        # Button to upload CSV file
        self.upload_btn = tk.Button(root, text="Upload CSV", command=self.upload_csv, width=20, bg="blue", fg="white")
        self.upload_btn.pack(pady=20)
        
        # Button to start model training and hyperparameter tuning
        self.train_btn = tk.Button(root, text="Start Training & Tuning", command=self.train_model, width=20, bg="green", fg="white")
        self.train_btn.pack(pady=20)

        # Label to display results
        self.result_label = tk.Label(root, text="", font=("Arial", 12))
        self.result_label.pack(pady=20)
    
    def upload_csv(self):
        # Let the user choose the dataset file
        self.filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.filepath:
            messagebox.showinfo("Success", "CSV file uploaded successfully!")
        else:
            messagebox.showwarning("Error", "No file selected!")

    def train_model(self):
        if not self.filepath:
            messagebox.showwarning("Error", "Please upload a CSV file first!")
            return
        
        # Load data and tune model
        X, y = load_data(self.filepath)
        best_params, accuracy = tune_model(X, y, model_type='logistic')
        
        # Display results
        result_text = f"Best Hyperparameters: {best_params}\nAccuracy: {accuracy:.4f}"
        self.result_label.config(text=result_text)

# Run the GUI application
if __name__ == "__main__":
    root = tk.Tk()
    app = TitanicModelApp(root)
    root.mainloop()
