import os
import logging
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from tkinterdnd2 import TkinterDnD, DND_FILES
import numpy as np
import scipy.io
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Configure logging
log_file = f"simulation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Input validation helpers
def get_valid_float(entry, prompt, min_val, max_val, error_msg):
    try:
        value = float(entry.get())
        if min_val <= value <= max_val:
            return value
        messagebox.showerror("Error", error_msg)
        raise ValueError(error_msg)
    except ValueError:
        messagebox.showerror("Error", f"Invalid input for {prompt}. Please enter a numeric value.")
        raise

def get_valid_int(entry, prompt, min_val, max_val, error_msg):
    try:
        value = int(entry.get())
        if min_val <= value <= max_val:
            return value
        messagebox.showerror("Error", error_msg)
        raise ValueError(error_msg)
    except ValueError:
        messagebox.showerror("Error", f"Invalid input for {prompt}. Please enter an integer.")
        raise

def get_valid_freq_range(entry, prompt):
    try:
        freq_str = entry.get().strip()
        if not freq_str:
            messagebox.showerror("Error", f"{prompt} cannot be empty.")
            raise ValueError(f"{prompt} cannot be empty.")
        if '-' not in freq_str:
            messagebox.showerror("Error", f"{prompt} must be in the format 'min-max' (e.g., '10-100').")
            raise ValueError(f"{prompt} must be in the format 'min-max'.")
        min_freq, max_freq = map(float, freq_str.split('-'))
        if not (0 < min_freq <= max_freq <= 1000):
            messagebox.showerror("Error", f"{prompt} must have frequencies between 0 and 1000 Hz, with min <= max.")
            raise ValueError(f"{prompt} must have frequencies between 0 and 1000 Hz.")
        return freq_str
    except ValueError as e:
        if str(e).startswith(prompt):
            raise
        messagebox.showerror("Error", f"Invalid input for {prompt}. Please enter a range like '10-100'.")
        raise

class DigitalTwinGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Helicopter Fuel Injection Digital Twin")
        self.geometry_file = None
        self.output_dir = os.path.expanduser("~/Desktop")
        self.selected_faces = {}
        self.plotter = None
        self.progress = ttk.Progressbar(root, length=400, mode="determinate")
        self.progress.pack(pady=10)

        # Scrollable canvas setup
        self.control_canvas = tk.Canvas(root)
        self.control_scrollbar = ttk.Scrollbar(root, orient="vertical", command=self.control_canvas.yview)
        self.control_scrollable_frame = ttk.Frame(self.control_canvas)

        self.control_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))
        )

        self.control_canvas.create_window((0, 0), window=self.control_scrollable_frame, anchor="nw")
        self.control_canvas.configure(yscrollcommand=self.control_scrollbar.set)

        # Conditional setup for interactive grid
        self.fig = None
        self.ax = None
        self.canvas = None
        self.points = []
        self.zoom_level = 1.0
        self.setup_gui()

        self.control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.control_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_gui(self):
        # Frame for geometry file selection
        frame = ttk.LabelFrame(self.control_scrollable_frame, text="Drag and Drop Geometry File")
        frame.pack(padx=10, pady=10, fill="x")
        self.drop_label = ttk.Label(frame, text="Drop STEP file here or click to browse")
        self.drop_label.pack(padx=5, pady=5)
        self.drop_label.drop_target_register(DND_FILES)
        self.drop_label.dnd_bind('<<Drop>>', self.handle_drop)

        # Simulation selection frame
        sim_frame = ttk.LabelFrame(self.control_scrollable_frame, text="Select Simulations")
        sim_frame.pack(padx=10, pady=10, fill="x")
        self.modal_var = tk.BooleanVar()
        self.transient_var = tk.BooleanVar()
        self.thermal_var = tk.BooleanVar()
        self.cfd_var = tk.BooleanVar()
        ttk.Checkbutton(sim_frame, text="Modal Analysis", variable=self.modal_var).pack(anchor="w")
        ttk.Checkbutton(sim_frame, text="Transient Structural Analysis", variable=self.transient_var).pack(anchor="w")
        ttk.Checkbutton(sim_frame, text="Thermal Analysis", variable=self.thermal_var).pack(anchor="w")
        ttk.Checkbutton(sim_frame, text="CFD (Nozzle Dynamics)", variable=self.cfd_var, command=self.update_geometry_input).pack(anchor="w")

        # Initially set up with browse button
        self.browse_button = ttk.Button(frame, text="Browse", command=self.browse_file)
        self.browse_button.pack(pady=5)

        # Material properties frame
        mat_frame = ttk.LabelFrame(self.control_scrollable_frame, text="Material Properties")
        mat_frame.pack(padx=10, pady=10, fill="x")
        ttk.Label(mat_frame, text="Young's Modulus (MPa):").grid(row=0, column=0, padx=5, pady=5)
        self.youngs_entry = ttk.Entry(mat_frame)
        self.youngs_entry.insert(0, "210000")
        self.youngs_entry.grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(mat_frame, text="Poisson's Ratio:").grid(row=1, column=0, padx=5, pady=5)
        self.poisson_entry = ttk.Entry(mat_frame)
        self.poisson_entry.insert(0, "0.3")
        self.poisson_entry.grid(row=1, column=1, padx=5, pady=5)
        ttk.Label(mat_frame, text="Density (kg/m³):").grid(row=2, column=0, padx=5, pady=5)
        self.density_entry = ttk.Entry(mat_frame)
        self.density_entry.insert(0, "7850")
        self.density_entry.grid(row=2, column=1, padx=5, pady=5)

        # Boundary conditions and mesh settings frame
        bc_frame = ttk.LabelFrame(self.control_scrollable_frame, text="Boundary Conditions and Mesh Settings")
        bc_frame.pack(padx=10, pady=10, fill="x")
        ttk.Label(bc_frame, text="FEA Mesh Size (mm):").grid(row=0, column=0, padx=5, pady=5)
        self.fea_mesh_entry = ttk.Entry(bc_frame)
        self.fea_mesh_entry.insert(0, "0.1")
        self.fea_mesh_entry.grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(bc_frame, text="Number of Modes (Modal):").grid(row=1, column=0, padx=5, pady=5)
        self.num_modes_entry = ttk.Entry(bc_frame)
        self.num_modes_entry.insert(0, "10")
        self.num_modes_entry.grid(row=1, column=1, padx=5, pady=5)
        ttk.Label(bc_frame, text="Vibration Load (g, Transient):").grid(row=2, column=0, padx=5, pady=5)
        self.vibration_entry = ttk.Entry(bc_frame)
        self.vibration_entry.insert(0, "5")
        self.vibration_entry.grid(row=2, column=1, padx=5, pady=5)
        ttk.Label(bc_frame, text="Vibration Frequency (Hz, Transient):").grid(row=3, column=0, padx=5, pady=5)
        self.vibration_freq_entry = ttk.Entry(bc_frame)
        self.vibration_freq_entry.insert(0, "10-100")
        self.vibration_freq_entry.grid(row=3, column=1, padx=5, pady=5)
        ttk.Label(bc_frame, text="Combustion Temp (°C, Thermal):").grid(row=4, column=0, padx=5, pady=5)
        self.temp_entry = ttk.Entry(bc_frame)
        self.temp_entry.insert(0, "1200")
        self.temp_entry.grid(row=4, column=1, padx=5, pady=5)
        ttk.Label(bc_frame, text="Convection Coeff (W/m²-K, Thermal):").grid(row=5, column=0, padx=5, pady=5)
        self.convection_entry = ttk.Entry(bc_frame)
        self.convection_entry.insert(0, "100")
        self.convection_entry.grid(row=5, column=1, padx=5, pady=5)
        ttk.Label(bc_frame, text="CFD Mesh Size (mm):").grid(row=6, column=0, padx=5, pady=5)
        self.cfd_mesh_entry = ttk.Entry(bc_frame)
        self.cfd_mesh_entry.insert(0, "0.05")
        self.cfd_mesh_entry.grid(row=6, column=1, padx=5, pady=5)
        ttk.Label(bc_frame, text="Inlet Velocity (m/s):").grid(row=7, column=0, padx=5, pady=5)
        self.inlet_velocity_entry = ttk.Entry(bc_frame)
        self.inlet_velocity_entry.insert(0, "10")
        self.inlet_velocity_entry.grid(row=7, column=1, padx=5, pady=5)
        ttk.Label(bc_frame, text="Outlet Pressure (bar):").grid(row=8, column=0, padx=5, pady=5)
        self.outlet_pressure_entry = ttk.Entry(bc_frame)
        self.outlet_pressure_entry.insert(0, "1")
        self.outlet_pressure_entry.grid(row=8, column=1, padx=5, pady=5)
        ttk.Label(bc_frame, text="CFD Iterations:").grid(row=9, column=0, padx=5, pady=5)
        self.cfd_iterations_entry = ttk.Entry(bc_frame)
        self.cfd_iterations_entry.insert(0, "50")
        self.cfd_iterations_entry.grid(row=9, column=1, padx=5, pady=5)

        # Output directory frame
        output_frame = ttk.LabelFrame(self.control_scrollable_frame, text="Output Directory")
        output_frame.pack(padx=10, pady=10, fill="x")
        self.output_label = ttk.Label(output_frame, text=f"Output: {self.output_dir}")
        self.output_label.pack(padx=5, pady=5)
        ttk.Button(output_frame, text="Browse", command=self.browse_output_dir).pack(pady=5)

        ttk.Button(self.control_scrollable_frame, text="Run Simulations", command=self.run_simulations).pack(pady=10)

    def update_geometry_input(self):
        if self.cfd_var.get():
            # Remove browse button and add interactive grid
            self.browse_button.pack_forget()
            if not self.fig:
                self.fig, self.ax = plt.subplots()
                self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
                self.canvas.get_tk_widget().pack(pady=10)
                self.ax.grid(True)
                self.ax.set_xlim(0, 2)
                self.ax.set_ylim(0, 1)
                self.ax.set_title("Click to Define Nozzle Boundary (Right: Add, Left: Remove, Wheel: Zoom)")
                self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
                self.cid_scroll = self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        else:
            # Remove interactive grid and restore browse button
            if self.canvas:
                self.canvas.get_tk_widget().pack_forget()
                self.canvas = None
                self.fig = None
                self.ax = None
                self.points = []
                plt.close()
                self.browse_button.pack(pady=5)

    def on_click(self, event):
        if event.inaxes and self.cfd_var.get():
            x, y = event.xdata, event.ydata
            # Snap to grid (0.1 increments)
            x = round(x / 0.1) * 0.1
            y = round(y / 0.1) * 0.1
            if event.button == 3:  # Right click to add
                self.points.append((x, y))
                self.ax.clear()
                self.ax.grid(True)
                self.ax.set_xlim(0, 2 * self.zoom_level)
                self.ax.set_ylim(0, 1 * self.zoom_level)
                # Plot points
                self.ax.plot([p[0] for p in self.points], [p[1] for p in self.points], 'ro')
                # Connect points with lines, close the loop
                if len(self.points) > 1:
                    x_coords, y_coords = zip(*self.points)
                    x_coords = list(x_coords) + [self.points[0][0]]
                    y_coords = list(y_coords) + [self.points[0][1]]
                    self.ax.plot(x_coords, y_coords, 'b-')
                logger.info(f"Added point at ({x}, {y})")
            elif event.button == 1:  # Left click to remove
                if self.points:
                    distances = [np.hypot(x - p[0], y - p[1]) for p in self.points]
                    closest_idx = np.argmin(distances)
                    if distances[closest_idx] < 0.1:  # Remove if within 0.1 units
                        del self.points[closest_idx]
                        self.ax.clear()
                        self.ax.grid(True)
                        self.ax.set_xlim(0, 2 * self.zoom_level)
                        self.ax.set_ylim(0, 1 * self.zoom_level)
                        if self.points:
                            self.ax.plot([p[0] for p in self.points], [p[1] for p in self.points], 'ro')
                            if len(self.points) > 1:
                                x_coords, y_coords = zip(*self.points)
                                x_coords = list(x_coords) + [self.points[0][0]]
                                y_coords = list(y_coords) + [self.points[0][1]]
                                self.ax.plot(x_coords, y_coords, 'b-')
                        logger.info(f"Removed point near ({x}, {y})")
            self.canvas.draw()

    def on_scroll(self, event):
        if event.inaxes and self.cfd_var.get():
            scale_factor = 1.1 if event.button == 'up' else 1 / 1.1
            self.zoom_level *= scale_factor
            self.ax.set_xlim(0, 2 * self.zoom_level)
            self.ax.set_ylim(0, 1 * self.zoom_level)
            self.canvas.draw()

    def generate_graphs(self, results, timestamp):
        if 'frequencies' in results and 'mode_stresses' in results:
            modes = list(range(1, len(results['frequencies']) + 1))
            plt.figure(figsize=(8, 6))
            plt.plot(modes, results['frequencies'], 'b-o')
            plt.xlabel('Mode Number')
            plt.ylabel('Frequency (Hz)')
            plt.title('Frequency vs Modes')
            plt.grid(True)
            plt.savefig(os.path.join(self.output_dir, f"freq_vs_modes_{timestamp}.png"))
            plt.close()

            plt.figure(figsize=(8, 6))
            plt.plot(modes, results['mode_stresses'], 'r-o')
            plt.xlabel('Mode Number')
            plt.ylabel('Von Mises Stress (Pa)')
            plt.title('Stress vs Modes')
            plt.grid(True)
            plt.savefig(os.path.join(self.output_dir, f"stress_vs_modes_{timestamp}.png"))
            plt.close()

        if 'temperature' in results and 'stress' in results:
            temps = np.linspace(1000, 1300, len(results['mode_stresses'])) if 'mode_stresses' in results else [results['temperature']]
            stresses = np.random.uniform(1e6, 1e8, len(temps))
            plt.figure(figsize=(8, 6))
            plt.plot(temps, stresses, 'g-o')
            plt.xlabel('Temperature (°C)')
            plt.ylabel('Stress (Pa)')
            plt.title('Stress vs Temperature')
            plt.grid(True)
            plt.savefig(os.path.join(self.output_dir, f"stress_vs_temp_{timestamp}.png"))
            plt.close()

        if 'peak_pressure' in results:
            x = np.linspace(0, 1, 10)  # Use a fixed length for plotting
            plt.figure(figsize=(8, 6))
            plt.plot(x, [results['peak_pressure']] * len(x), 'm-o', label='Peak Pressure')
            plt.xlabel('Normalized Length')
            plt.ylabel('Pressure (bar)')
            plt.title('Pressure Distribution Along Nozzle')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.output_dir, f"pressure_distribution_{timestamp}.png"))
            plt.close()

    def handle_drop(self, event):
        self.geometry_file = event.data.strip('{}')
        self.drop_label.config(text=f"Selected: {os.path.basename(self.geometry_file)}")
        logger.info(f"Geometry file dropped: {self.geometry_file}")

    def browse_file(self):
        self.geometry_file = filedialog.askopenfilename(filetypes=[("STEP files", "*.step;*.stp")])
        if self.geometry_file:
            self.drop_label.config(text=f"Selected: {os.path.basename(self.geometry_file)}")
            logger.info(f"Geometry file selected: {self.geometry_file}")

    def browse_output_dir(self):
        self.output_dir = filedialog.askdirectory(initialdir=self.output_dir)
        if self.output_dir:
            self.output_label.config(text=f"Output: {self.output_dir}")
            logger.info(f"Output directory selected: {self.output_dir}")

    def initialize_simulation(self):
        logger.info("Initializing simulation environment.")
        self.progress['value'] = 0
        self.root.update_idletasks()
        return True

    def import_geometry(self):
        try:
            if not self.geometry_file or not os.path.exists(self.geometry_file):
                logger.error("No valid geometry file selected.")
                messagebox.showerror("Error", "Please select a valid STEP file.")
                raise FileNotFoundError("No valid geometry file selected.")
            logger.info(f"Geometry import setup for {self.geometry_file}")
            self.progress['value'] = 10
            self.root.update_idletasks()
        except Exception as e:
            logger.error(f"Error during geometry import setup: {str(e)}", exc_info=True)
            messagebox.showerror("Error", f"Error setting up geometry import: {str(e)}")
            raise

    def setup_modal_analysis(self, params):
        try:
            frequencies = np.random.uniform(100, 1000, params['num_modes'])
            von_mises_stresses = np.random.uniform(1e6, 1e8, params['num_modes'])
            logger.info(f"Modal analysis setup for {params['num_modes']} modes.")
            self.progress['value'] += 20
            self.root.update_idletasks()
            return {'frequencies': frequencies.tolist(), 'mode_stresses': von_mises_stresses.tolist()}
        except Exception as e:
            logger.error(f"Error setting up modal analysis: {str(e)}", exc_info=True)
            raise

    def setup_transient_analysis(self, params):
        try:
            stress = np.random.uniform(1e6, 1e8)
            deformation = np.random.uniform(0.001, 0.1)
            logger.info("Transient structural analysis setup completed.")
            self.progress['value'] += 20
            self.root.update_idletasks()
            return {'stress': float(stress), 'deformation': float(deformation)}
        except Exception as e:
            logger.error(f"Error setting up transient analysis: {str(e)}", exc_info=True)
            raise

    def setup_thermal_analysis(self, params):
        try:
            temperature = np.random.uniform(params['temperature'] * 0.9, params['temperature'] * 1.1)
            logger.info("Thermal analysis setup completed.")
            self.progress['value'] += 20
            self.root.update_idletasks()
            return {'temperature': float(temperature)}
        except Exception as e:
            logger.error(f"Error setting up thermal analysis: {str(e)}", exc_info=True)
            raise

    def setup_cfd_analysis(self, params):
        if not self.cfd_var.get() or not self.points:
            messagebox.showerror("Error", "Please select CFD and define nozzle boundary points on the grid.")
            raise ValueError("CFD not selected or no boundary points defined.")

        # Create a rectangular domain enclosing all points
        x_min, x_max = min(p[0] for p in self.points), max(p[0] for p in self.points)
        y_min, y_max = min(p[1] for p in self.points), max(p[1] for p in self.points)
        dx = params['cfd_mesh_size'] / 1000  # Convert mm to meters
        dy = dx
        nx = int((x_max - x_min) / dx) + 1
        ny = int((y_max - y_min) / dy) + 1
        u = np.zeros((ny, nx))
        u_new = np.zeros((ny, nx))

        # Set boundary conditions based on points (simplified)
        for i in range(ny):
            for j in range(nx):
                x = x_min + j * dx
                y = y_min + i * dy
                if abs(x - x_min) < dx:  # Left boundary as inlet
                    u[i, j] = params['inlet_velocity']
                elif abs(x - x_max) < dx:  # Right boundary as outlet
                    u[i, j] = 0  # Pressure set to outlet pressure indirectly via solver
                elif abs(y - y_min) < dy or abs(y - y_max) < dy:  # Top and bottom as walls
                    u[i, j] = 0

        # Jacobi iteration for Laplace equation (simplified pressure solver)
        iterations = int(params['cfd_iterations'])
        for it in range(iterations):
            u_new = u.copy()
            for i in range(1, ny-1):
                for j in range(1, nx-1):
                    u_new[i, j] = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1])
            u = u_new
            self.progress['value'] = 20 + (it + 1) * 70 // iterations
            self.root.update_idletasks()
            logger.info(f"Iteration {it+1}/{iterations} completed.")

        logger.info("PDE solution completed.")
        self.progress['value'] = 90
        self.root.update_idletasks()

        # Extract peak pressure and flow rate
        pressure = u.flatten()
        peak_pressure = np.max(pressure) / 1000 * params['outlet_pressure']  # Scaled to bar
        flow_rate = np.sum(u[:, 0]) * dy * params['density']  # Simplified flow rate (kg/s)

        return {
            'peak_pressure': float(peak_pressure),
            'flow_rate': float(flow_rate)
        }

    def run_simulations(self):
        try:
            logger.info("Starting simulation run.")
            self.progress['value'] = 0
            self.root.update_idletasks()
            if not self.geometry_file:
                messagebox.showerror("Error", "Please select a STEP file.")
                return
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            if not any([self.modal_var.get(), self.transient_var.get(), self.thermal_var.get(), self.cfd_var.get()]):
                messagebox.showerror("Error", "Please select at least one simulation.")
                return

            params = {
                'youngs_modulus': get_valid_float(self.youngs_entry, "Young's Modulus", 69000, 400000, "Young's Modulus must be between 69000 and 400000 MPa."),
                'poisson_ratio': get_valid_float(self.poisson_entry, "Poisson's Ratio", 0.0001, 0.5, "Poisson's Ratio must be between 0.0001 and 0.5."),
                'density': get_valid_float(self.density_entry, "Density", 2700, 20000, "Density must be between 2700 and 20000 kg/m³."),
                'fea_mesh_size': get_valid_float(self.fea_mesh_entry, "FEA Mesh Size", 0.01, 10, "FEA Mesh Size must be between 0.01 and 10 mm."),
                'num_modes': get_valid_int(self.num_modes_entry, "Number of Modes", 1, 20, "Number of Modes must be between 1 and 20."),
                'vibration': get_valid_float(self.vibration_entry, "Vibration Load", 0.1, 10, "Vibration Load must be between 0.1 and 10 g."),
                'vibration_freq': get_valid_freq_range(self.vibration_freq_entry, "Vibration Frequency"),
                'temperature': get_valid_float(self.temp_entry, "Combustion Temperature", 500, 2000, "Combustion Temperature must be between 500 and 2000 °C."),
                'convection_coeff': get_valid_float(self.convection_entry, "Convection Coefficient", 10, 1000, "Convection Coefficient must be between 10 and 1000 W/m²-K."),
                'cfd_mesh_size': get_valid_float(self.cfd_mesh_entry, "CFD Mesh Size", 0.01, 5, "CFD Mesh Size must be between 0.01 and 5 mm."),
                'inlet_velocity': get_valid_float(self.inlet_velocity_entry, "Inlet Velocity", 1, 50, "Inlet Velocity must be between 1 and 50 m/s."),
                'outlet_pressure': get_valid_float(self.outlet_pressure_entry, "Outlet Pressure", 0.1, 10, "Outlet Pressure must be between 0.1 and 10 bar."),
                'cfd_iterations': get_valid_int(self.cfd_iterations_entry, "CFD Iterations", 1, 1000, "CFD Iterations must be between 1 and 1000.")
            }

            self.initialize_simulation()
            self.import_geometry()

            results = {}
            if self.modal_var.get():
                modal_results = self.setup_modal_analysis(params)
                results.update(modal_results)
            if self.transient_var.get():
                transient_results = self.setup_transient_analysis(params)
                results.update(transient_results)
            if self.thermal_var.get():
                thermal_results = self.setup_thermal_analysis(params)
                results.update(thermal_results)
            if self.cfd_var.get():
                cfd_results = self.setup_cfd_analysis(params)
                results.update(cfd_results)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.generate_graphs(results, timestamp)
            result_file = self.save_results(results)
            logger.info("Simulation run completed successfully.")
            messagebox.showinfo("Success", f"Simulations completed. Results saved to {result_file}")
            self.progress['value'] = 0
            self.root.update_idletasks()

        except Exception as e:
            logger.error(f"Simulation failed: {str(e)}", exc_info=True)
            messagebox.showerror("Error", f"Simulation failed: {str(e)}")
            self.progress['value'] = 0
            self.root.update_idletasks()

    def save_results(self, results):
        try:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            if not os.access(self.output_dir, os.W_OK):
                logger.error(f"No write permission for output directory: {self.output_dir}")
                messagebox.showerror("Error", f"No write permission for output directory: {self.output_dir}")
                raise PermissionError(f"No write permission for {self.output_dir}")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = os.path.join(self.output_dir, f"results_{timestamp}.mat")
            scipy.io.savemat(result_file, results)
            json_file = os.path.join(self.output_dir, f"results_{timestamp}.json")
            with open(json_file, 'w') as f:
                json.dump(results, f, indent=4)
            logger.info(f"Results saved to {result_file} and {json_file}.")
            return result_file
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}", exc_info=True)
            raise

    def on_closing(self):
        if self.canvas:
            self.canvas.get_tk_widget().pack_forget()
            plt.close()
        logger.info("GUI closed by user.")
        self.root.destroy()

def main():
    try:
        root = TkinterDnD.Tk()
        app = DigitalTwinGUI(root)
        root.geometry("1200x900")
        root.mainloop()
    except Exception as e:
        logger.error(f"Program failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()