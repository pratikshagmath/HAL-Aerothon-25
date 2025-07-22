import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import scrolledtext  # For scrolling

# Step 1: Load the Dataset
dataset_path = r'fuel_system_dataset.csv'
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset file '{dataset_path}' not found in {os.getcwd()}. This should not happen as you confirmed the file exists.")
df = pd.read_csv(dataset_path)
print(f"Dataset loaded from '{dataset_path}'")

# Step 2: Preprocess the data
X = df[['throttle_deflection_percent', 'tank_pressure_bar', 'pump_pressure_bar',
        'filter_pressure_drop_bar', 'valve_opening_percent', 'injector_flow_lpm',
        'fuel_temperature_c', 'rotor_speed_rpm', 'vibration_mm_s']]

le = LabelEncoder()
y = le.fit_transform(df['health_status'])

# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Initial Model Accuracy: {accuracy:.2f}")
print("\nInitial Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')  # Fallback to save if display fails
plt.show()

# Save Feature Importance
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance.sort_values(ascending=False).plot(kind='bar', figsize=(10, 6))
plt.title('Feature Importance')
plt.ylabel('Importance')
plt.savefig('feature_importance.png')  # Fallback to save if display fails
plt.show()

# Step 6: Test predictions on sample data
sample_data = np.array([
    [50.0, 0.7, 80.0, 5.0, 50.0, 20.0, 15.0, 6000.0, 2.0],  # Healthy
    [70.0, 0.7, 20.0, 10.0, 70.0, 15.0, 15.0, 5800.0, 8.0],  # Critical (Pump Failure)
    [80.0, 0.7, 95.0, 25.0, 80.0, 30.0, 15.0, 5900.0, 3.0],  # Warning (Clogged Filter)
])
predictions = model.predict(sample_data)
predicted_labels = le.inverse_transform(predictions)
for i, pred in enumerate(predicted_labels):
    print(f"Sample {i+1}: Health Status = {pred}")

# Step 7: Save the model
joblib.dump(model, 'ardiden_health_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
print("\nModel and label encoder saved successfully.")

# GUI Functions
features = ['throttle_deflection_percent', 'tank_pressure_bar', 'pump_pressure_bar',
            'filter_pressure_drop_bar', 'valve_opening_percent', 'injector_flow_lpm',
            'fuel_temperature_c', 'rotor_speed_rpm', 'vibration_mm_s']

def create_input_gui():
    input_window = tk.Tk()
    input_window.title("Ardiden 1H1 Health Monitor - Input")
    input_window.geometry("400x500")
    input_window.configure(bg='#e6f3ff')  # Light blue background

    # Create a scrollable frame
    canvas = tk.Canvas(input_window, bg='#e6f3ff')
    scrollbar = ttk.Scrollbar(input_window, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    # Pack the canvas and scrollbar
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Title
    title_label = tk.Label(scrollable_frame, text="Enter Sensor Values", font=('Arial', 16, 'bold'), bg='#e6f3ff', fg='#003366')
    title_label.pack(pady=10)

    # Entry fields
    entries = {}
    for i, feature in enumerate(features):
        label = tk.Label(scrollable_frame, text=f"{feature.replace('_', ' ').title()}:", font=('Arial', 10), bg='#e6f3ff', fg='#003366')
        label.pack(pady=5)
        entry = tk.Entry(scrollable_frame, width=20, font=('Arial', 10))
        entry.pack(pady=5)
        entries[feature] = entry

    # Submit button
    def on_submit():
        try:
            input_data = [float(entries[feat].get()) for feat in features]
            if any(v is None for v in input_data):
                raise ValueError("All fields must be filled.")
            input_window.destroy()
            create_output_gui(input_data)
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))

    submit_button = tk.Button(scrollable_frame, text="Submit", command=on_submit, font=('Arial', 12, 'bold'), bg='#0066cc', fg='white', activebackground='#004d99')
    submit_button.pack(pady=20)

    input_window.mainloop()

def create_output_gui(input_data):
    output_window = tk.Tk()
    output_window.title("Ardiden 1H1 Health Monitor - Result")
    output_window.geometry("400x300")
    output_window.configure(bg='#e6f3ff')

    # Predict
    input_array = np.array([input_data])
    prediction = model.predict(input_array)
    predicted_label = le.inverse_transform(prediction)[0]
    failure_mode = 'None'  # Default

    # Determine background color and failure mode based on prediction
    if predicted_label == 'Healthy':
        bg_color = '#90ee90'  # Light green
    elif predicted_label == 'Warning':
        bg_color = '#ffff99'  # Light yellow
        if input_data[3] > 20:  # filter_pressure_drop_bar
            failure_mode = 'Clogged Filter'
        elif input_data[5] < 25:  # injector_flow_lpm
            failure_mode = 'Injector Clogging'
    else:  # Critical
        bg_color = '#ff9999'  # Light red
        if input_data[2] < 40:  # pump_pressure_bar
            failure_mode = 'Pump Failure'
        elif input_data[4] == 50.0:  # valve_opening_percent
            failure_mode = 'Valve Sticking'

    output_window.configure(bg=bg_color)

    # Title
    title_label = tk.Label(output_window, text="Health Status", font=('Arial', 16, 'bold'), bg=bg_color, fg='#003366')
    title_label.pack(pady=10)

    # Result
    result_label = tk.Label(output_window, text=f"Status: {predicted_label}", font=('Arial', 14), bg=bg_color, fg='#003366')
    result_label.pack(pady=10)

    failure_label = tk.Label(output_window, text=f"Possible Failure: {failure_mode}", font=('Arial', 12), bg=bg_color, fg='#003366')
    failure_label.pack(pady=10)

    # Back button
    def on_back():
        output_window.destroy()
        create_input_gui()

    back_button = tk.Button(output_window, text="Back", command=on_back, font=('Arial', 12, 'bold'), bg='#0066cc', fg='white', activebackground='#004d99')
    back_button.pack(pady=20)

    output_window.mainloop()

# Start the GUI after training
if __name__ == "__main__":
    print("Training model and preparing GUI...")
    create_input_gui()