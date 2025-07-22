import os
import json
import logging
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from tkinterdnd2 import TkinterDnD, DND_FILES
import numpy as np
import math

# Configure logging
log_file = f"cockpit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class HelicopterCockpitUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Helicopter Cockpit Simulator")
        self.root.geometry("1200x800")
        self.ansys_data = {}
        self.simulink_data = {}
        self.flight_params = {
            'attitude': 0.0,  # degrees
            'altitude': 0.0,  # meters
            'rpm': 300.0,     # RPM
            'fuel_volume': 100.0  # liters
        }

        # Canvas for cockpit display
        self.canvas = tk.Canvas(root, bg="black", height=800, width=1200)
        self.canvas.pack()

        # Setup drag-and-drop areas
        self.setup_drag_drop()
        self.draw_cockpit()
        self.update_display()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_drag_drop(self):
        # Ansys Drop Area
        ansys_frame = ttk.LabelFrame(self.root, text="Drop Ansys Data (JSON)")
        ansys_frame.place(x=10, y=10, width=300, height=100)
        self.ansys_label = tk.Label(ansys_frame, text="Drop Ansys file here", bg="lightgray")
        self.ansys_label.pack(fill="both", expand=True)
        self.ansys_label.drop_target_register(DND_FILES)
        self.ansys_label.dnd_bind('<<Drop>>', lambda e: self.handle_drop(e, "ansys"))

        # Simulink Drop Area
        simulink_frame = ttk.LabelFrame(self.root, text="Drop Simulink Data (JSON)")
        simulink_frame.place(x=10, y=120, width=300, height=100)
        self.simulink_label = tk.Label(simulink_frame, text="Drop Simulink file here", bg="lightgray")
        self.simulink_label.pack(fill="both", expand=True)
        self.simulink_label.drop_target_register(DND_FILES)
        self.simulink_label.dnd_bind('<<Drop>>', lambda e: self.handle_drop(e, "simulink"))

    def handle_drop(self, event, source):
        """Handle drag-and-drop for Ansys or Simulink data."""
        file_path = event.data.strip('{}')
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            if source == "ansys":
                self.ansys_data = data
                self.ansys_label.config(text=f"Loaded: {os.path.basename(file_path)}")
                logger.info(f"Ansys data loaded from {file_path}")
            elif source == "simulink":
                self.simulink_data = data
                self.simulink_label.config(text=f"Loaded: {os.path.basename(file_path)}")
                logger.info(f"Simulink data loaded from {file_path}")
            self.calculate_flight_params()
            self.update_display()
        except Exception as e:
            logger.error(f"Error loading {source} data: {str(e)}")
            messagebox.showerror("Error", f"Failed to load {source} data: {str(e)}")

    def calculate_flight_params(self):
        """Calculate flight parameters based on Ansys and Simulink data."""
        if self.ansys_data and self.simulink_data:
            # Extract mocked data (replace with actual keys from your JSON)
            pressure_drop = self.ansys_data.get('pressure_drop', 0.3)  # bar
            flow_velocity = self.ansys_data.get('flow_velocity', 30.0)  # m/s
            fuel_flow = self.simulink_data.get('fuel_flow', 110.0)  # l/hr
            efficiency = self.simulink_data.get('efficiency', 0.9)
            throttle = self.simulink_data.get('vibration', 0.5)  # Using vibration as proxy for throttle effect

            # Calculations
            self.flight_params['attitude'] = (pressure_drop * 10) - 2  # Simplified attitude adjustment (degrees)
            self.flight_params['altitude'] = (flow_velocity * 5) + (throttle * 50)  # meters
            self.flight_params['rpm'] = 250 + (efficiency * 100)  # RPM
            self.flight_params['fuel_volume'] = max(0, self.flight_params['fuel_volume'] - (fuel_flow / 3600))  # liters, per second

    def draw_cockpit(self):
        """Draw the cockpit-style display."""
        # Background and borders
        self.canvas.create_rectangle(320, 50, 1180, 750, fill="darkgray", outline="white")

        # Attitude Indicator
        self.canvas.create_oval(350, 100, 550, 300, fill="black")
        self.attitude_needle = self.canvas.create_line(450, 200, 450, 150, fill="green", width=2)

        # Altimeter
        self.canvas.create_rectangle(600, 100, 800, 300, fill="black")
        self.altitude_text = self.canvas.create_text(700, 200, text="0 m", fill="white", font=("Arial", 20))

        # RPM Gauge
        self.canvas.create_oval(850, 100, 1050, 300, fill="black")
        self.rpm_text = self.canvas.create_text(950, 200, text="300 RPM", fill="white", font=("Arial", 20))
        self.rpm_needle = self.canvas.create_line(950, 200, 950, 150, fill="red", width=2)

        # Fuel Gauge
        self.canvas.create_rectangle(600, 350, 800, 550, fill="black")
        self.fuel_text = self.canvas.create_text(700, 450, text="100 L", fill="white", font=("Arial", 20))
        self.fuel_bar = self.canvas.create_rectangle(650, 400, 750, 500, fill="green")

        # Horizon Line
        self.horizon = self.canvas.create_line(320, 400, 1180, 400, fill="blue", width=2)

        # Labels
        self.canvas.create_text(450, 50, text="Attitude", fill="white", font=("Arial", 14))
        self.canvas.create_text(700, 50, text="Altitude", fill="white", font=("Arial", 14))
        self.canvas.create_text(950, 50, text="RPM", fill="white", font=("Arial", 14))
        self.canvas.create_text(700, 300, text="Fuel", fill="white", font=("Arial", 14))

    def update_display(self):
        """Update the cockpit display with current flight parameters."""
        attitude = self.flight_params['attitude']
        altitude = self.flight_params['altitude']
        rpm = self.flight_params['rpm']
        fuel = self.flight_params['fuel_volume']

        # Update Attitude Indicator
        angle = -attitude * math.pi / 180
        x2 = 450 + 50 * math.sin(angle)
        y2 = 200 - 50 * math.cos(angle)
        self.canvas.coords(self.attitude_needle, 450, 200, x2, y2)

        # Update Altimeter
        self.canvas.itemconfig(self.altitude_text, text=f"{int(altitude)} m")

        # Update RPM Gauge
        angle = (rpm - 250) / 100 * 0.5 * math.pi - math.pi / 2
        x2 = 950 + 50 * math.cos(angle)
        y2 = 200 + 50 * math.sin(angle)
        self.canvas.coords(self.rpm_needle, 950, 200, x2, y2)
        self.canvas.itemconfig(self.rpm_text, text=f"{int(rpm)} RPM")

        # Update Fuel Gauge
        fuel_height = 100 * (fuel / 100)
        self.canvas.coords(self.fuel_bar, 650, 500 - fuel_height, 750, 500)
        self.canvas.itemconfig(self.fuel_text, text=f"{int(fuel)} L")

        # Update Horizon
        horizon_y = 400 - (attitude * 2)
        self.canvas.coords(self.horizon, 320, horizon_y, 1180, horizon_y)

        self.root.after(1000, self.update_display)  # Update every second

    def on_closing(self):
        """Handle window close event."""
        logger.info("Cockpit UI closed by user.")
        self.root.destroy()

def main():
    """Main function to launch the UI."""
    try:
        root = TkinterDnD.Tk()
        app = HelicopterCockpitUI(root)
        root.mainloop()
    except Exception as e:
        logger.error(f"Program failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()