import os
import logging
import uuid
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from tkinterdnd2 import TkinterDnD, DND_FILES
import numpy as np
import matlab.engine
import scipy.io

# Configure logging
log_file = f"simulation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Input validation helpers
def get_valid_float(entry, prompt, min_val, max_val, error_msg):
    """Validate float input from GUI entry."""
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
    """Validate integer input from GUI entry."""
    try:
        value = int(entry.get())
        if min_val <= value <= max_val:
            return value
        messagebox.showerror("Error", error_msg)
        raise ValueError(error_msg)
    except ValueError:
        messagebox.showerror("Error", f"Invalid input for {prompt}. Please enter an integer.")
        raise

class DigitalTwinGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Helicopter Fuel Injection Digital Twin")
        self.geometry_file = None
        self.output_dir = os.path.expanduser("~/Desktop")
        self.eng = None  # MATLAB engine

        # GUI Layout
        self.setup_gui()

    def setup_gui(self):
        """Set up the GUI layout."""
        # Drag and Drop Frame
        frame = ttk.LabelFrame(self.root, text="Drag and Drop CATIA STEP File")
        frame.pack(padx=10, pady=10, fill="x")
        self.drop_label = ttk.Label(frame, text="Drop STEP file here or click to browse")
        self.drop_label.pack(padx=5, pady=5)
        self.drop_label.drop_target_register(DND_FILES)
        self.drop_label.dnd_bind('<<Drop>>', self.handle_drop)
        ttk.Button(frame, text="Browse", command=self.browse_file).pack(pady=5)

        # Simulation Selection
        sim_frame = ttk.LabelFrame(self.root, text="Select Simulations")
        sim_frame.pack(padx=10, pady=10, fill="x")
        self.modal_var = tk.BooleanVar()
        self.transient_var = tk.BooleanVar()
        self.thermal_var = tk.BooleanVar()
        self.cfd_var = tk.BooleanVar()
        ttk.Checkbutton(sim_frame, text="Modal Analysis", variable=self.modal_var).pack(anchor="w")
        ttk.Checkbutton(sim_frame, text="Transient Structural Analysis", variable=self.transient_var).pack(anchor="w")
        ttk.Checkbutton(sim_frame, text="Thermal Analysis", variable=self.thermal_var).pack(anchor="w")
        ttk.Checkbutton(sim_frame, text="CFD (Nozzle Dynamics)", variable=self.cfd_var).pack(anchor="w")

        # Material Properties
        mat_frame = ttk.LabelFrame(self.root, text="Material Properties")
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

        # Boundary Conditions and Mesh Settings
        bc_frame = ttk.LabelFrame(self.root, text="Boundary Conditions and Mesh Settings")
        bc_frame.pack(padx=10, pady=10, fill="x")

        # FEA Settings
        ttk.Label(bc_frame, text="FEA Mesh Size (mm):").grid(row=0, column=0, padx=5, pady=5)
        self.fea_mesh_entry = ttk.Entry(bc_frame)
        self.fea_mesh_entry.insert(0, "0.1")
        self.fea_mesh_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(bc_frame, text="Number of Modes (Modal):").grid(row=1, column=0, padx=5, pady=5)
        self.num_modes_entry = ttk.Entry(bc_frame)
        self.num_modes_entry.insert(0, "10")
        self.num_modes_entry.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(bc_frame, text="Fixed Support (Face ID):").grid(row=2, column=0, padx=5, pady=5)
        self.fixed_support_entry = ttk.Entry(bc_frame)
        self.fixed_support_entry.insert(0, "BaseFace")
        self.fixed_support_entry.grid(row=2, column=1, padx=5, pady=5)

        ttk.Label(bc_frame, text="Vibration Load (g, Transient):").grid(row=3, column=0, padx=5, pady=5)
        self.vibration_entry = ttk.Entry(bc_frame)
        self.vibration_entry.insert(0, "5")
        self.vibration_entry.grid(row=3, column=1, padx=5, pady=5)

        ttk.Label(bc_frame, text="Vibration Frequency (Hz, Transient):").grid(row=4, column=0, padx=5, pady=5)
        self.vibration_freq_entry = ttk.Entry(bc_frame)
        self.vibration_freq_entry.insert(0, "10-100")
        self.vibration_freq_entry.grid(row=4, column=1, padx=5, pady=5)

        ttk.Label(bc_frame, text="Combustion Temp (°C, Thermal):").grid(row=5, column=0, padx=5, pady=5)
        self.temp_entry = ttk.Entry(bc_frame)
        self.temp_entry.insert(0, "1200")
        self.temp_entry.grid(row=5, column=1, padx=5, pady=5)

        ttk.Label(bc_frame, text="Convection Coeff (W/m²-K, Thermal):").grid(row=6, column=0, padx=5, pady=5)
        self.convection_entry = ttk.Entry(bc_frame)
        self.convection_entry.insert(0, "100")
        self.convection_entry.grid(row=6, column=1, padx=5, pady=5)

        # CFD Settings
        ttk.Label(bc_frame, text="CFD Mesh Size (mm):").grid(row=7, column=0, padx=5, pady=5)
        self.cfd_mesh_entry = ttk.Entry(bc_frame)
        self.cfd_mesh_entry.insert(0, "0.05")
        self.cfd_mesh_entry.grid(row=7, column=1, padx=5, pady=5)

        ttk.Label(bc_frame, text="Inlet Pressure (bar):").grid(row=8, column=0, padx=5, pady=5)
        self.pressure_entry = ttk.Entry(bc_frame)
        self.pressure_entry.insert(0, "150")
        self.pressure_entry.grid(row=8, column=1, padx=5, pady=5)

        ttk.Label(bc_frame, text="Outlet Pressure (bar):").grid(row=9, column=0, padx=5, pady=5)
        self.outlet_pressure_entry = ttk.Entry(bc_frame)
        self.outlet_pressure_entry.insert(0, "1")
        self.outlet_pressure_entry.grid(row=9, column=1, padx=5, pady=5)

        ttk.Label(bc_frame, text="Turbulence Model:").grid(row=10, column=0, padx=5, pady=5)
        self.turbulence_model = ttk.Combobox(bc_frame, values=["k-epsilon", "k-omega", "LES"])
        self.turbulence_model.set("k-epsilon")
        self.turbulence_model.grid(row=10, column=1, padx=5, pady=5)

        ttk.Label(bc_frame, text="Reverse Flow:").grid(row=11, column=0, padx=5, pady=5)
        self.reverse_flow_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(bc_frame, variable=self.reverse_flow_var).grid(row=11, column=1, padx=5, pady=5)

        ttk.Label(bc_frame, text="CFD Iterations:").grid(row=12, column=0, padx=5, pady=5)
        self.cfd_iterations_entry = ttk.Entry(bc_frame)
        self.cfd_iterations_entry.insert(0, "100")
        self.cfd_iterations_entry.grid(row=12, column=1, padx=5, pady=5)

        # Output Directory
        output_frame = ttk.LabelFrame(self.root, text="Output Directory")
        output_frame.pack(padx=10, pady=10, fill="x")
        self.output_label = ttk.Label(output_frame, text=f"Output: {self.output_dir}")
        self.output_label.pack(padx=5, pady=5)
        ttk.Button(output_frame, text="Browse", command=self.browse_output_dir).pack(pady=5)

        # Run Button
        ttk.Button(self.root, text="Run Simulations", command=self.run_simulations).pack(pady=10)

    def handle_drop(self, event):
        """Handle drag-and-drop file event."""
        self.geometry_file = event.data.strip('{}')
        self.drop_label.config(text=f"Selected: {os.path.basename(self.geometry_file)}")
        logger.info(f"Geometry file dropped: {self.geometry_file}")

    def browse_file(self):
        """Browse for STEP file."""
        self.geometry_file = filedialog.askopenfilename(filetypes=[("STEP files", "*.step;*.stp")])
        if self.geometry_file:
            self.drop_label.config(text=f"Selected: {os.path.basename(self.geometry_file)}")
            logger.info(f"Geometry file selected: {self.geometry_file}")

    def browse_output_dir(self):
        """Browse for output directory."""
        self.output_dir = filedialog.askdirectory(initialdir=self.output_dir)
        if self.output_dir:
            self.output_label.config(text=f"Output: {self.output_dir}")
            logger.info(f"Output directory selected: {self.output_dir}")

    def initialize_simulation(self):
        """Initialize simulation environment (mocked for Ansys)."""
        logger.info("Simulation environment initialized (mocked).")
        return True

    def import_geometry(self):
        """Import CATIA geometry (STEP file) - mocked."""
        try:
            if not self.geometry_file or not os.path.exists(self.geometry_file):
                logger.error("No valid geometry file selected.")
                messagebox.showerror("Error", "Please select a valid STEP file.")
                raise FileNotFoundError("No valid geometry file selected.")
            logger.info(f"Imported geometry from {self.geometry_file} (mocked).")
        except Exception as e:
            logger.error(f"Error importing geometry: {str(e)}", exc_info=True)
            messagebox.showerror("Error", f"Error importing geometry: {str(e)}")
            raise

    def setup_modal_analysis(self, params):
        """Set up modal analysis (mocked)."""
        try:
            frequencies = np.random.uniform(100, 1000, params['num_modes'])
            von_mises_stresses = np.random.uniform(1e6, 1e8, params['num_modes'])
            logger.info(f"Modal analysis setup for {params['num_modes']} modes (mocked).")
            return {'frequencies': frequencies, 'mode_stresses': von_mises_stresses}
        except Exception as e:
            logger.error(f"Error setting up modal analysis: {str(e)}", exc_info=True)
            raise

    def setup_transient_analysis(self, params):
        """Set up transient structural analysis (mocked)."""
        try:
            stress = np.random.uniform(1e6, 1e8)
            deformation = np.random.uniform(0.001, 0.1)
            logger.info("Transient structural analysis setup completed (mocked).")
            return {'stress': stress, 'deformation': deformation}
        except Exception as e:
            logger.error(f"Error setting up transient analysis: {str(e)}", exc_info=True)
            raise

    def setup_thermal_analysis(self, params):
        """Set up thermal analysis (mocked)."""
        try:
            temperature = np.random.uniform(params['temperature'] * 0.9, params['temperature'] * 1.1)
            logger.info("Thermal analysis setup completed (mocked).")
            return {'temperature': temperature}
        except Exception as e:
            logger.error(f"Error setting up thermal analysis: {str(e)}", exc_info=True)
            raise

    def setup_cfd_analysis(self, params):
        """Set up CFD analysis (mocked)."""
        try:
            flow_rate = np.random.uniform(0.1, 1.0)
            reverse_flow = np.random.uniform(-0.1, 0.1) if params['reverse_flow'] else 0.0
            logger.info(f"CFD analysis setup completed with {params['cfd_iterations']} iterations (mocked).")
            return {'flow_rate': flow_rate, 'reverse_flow': reverse_flow}
        except Exception as e:
            logger.error(f"Error setting up CFD analysis: {str(e)}", exc_info=True)
            raise

    def run_simulations(self):
        """Run selected simulations and save results."""
        try:
            if not self.geometry_file:
                messagebox.showerror("Error", "Please select a STEP file.")
                return
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
                logger.info(f"Created output directory: {self.output_dir}")
            if not any([self.modal_var.get(), self.transient_var.get(), self.thermal_var.get(), self.cfd_var.get()]):
                messagebox.showerror("Error", "Please select at least one simulation.")
                return

            # Validate inputs
            params = {
                'youngs_modulus': get_valid_float(self.youngs_entry, "Young's Modulus", 69000, 400000, "Young's Modulus must be between 69000 and 400000 MPa."),
                'poisson_ratio': get_valid_float(self.poisson_entry, "Poisson's Ratio", 0.0001, 0.5, "Poisson's Ratio must be between 0.0001 and 0.5."),
                'density': get_valid_float(self.density_entry, "Density", 2700, 20000, "Density must be between 2700 and 20000 kg/m³."),
                'fea_mesh_size': get_valid_float(self.fea_mesh_entry, "FEA Mesh Size", 0.01, 10, "FEA Mesh Size must be between 0.01 and 10 mm."),
                'num_modes': get_valid_int(self.num_modes_entry, "Number of Modes", 1, 20, "Number of Modes must be between 1 and 20."),
                'fixed_support': self.fixed_support_entry.get(),
                'vibration': get_valid_float(self.vibration_entry, "Vibration Load", 0.1, 10, "Vibration Load must be between 0.1 and 10 g."),
                'vibration_freq': self.vibration_freq_entry.get(),
                'temperature': get_valid_float(self.temp_entry, "Combustion Temperature", 500, 2000, "Combustion Temperature must be between 500 and 2000 °C."),
                'convection_coeff': get_valid_float(self.convection_entry, "Convection Coefficient", 10, 1000, "Convection Coefficient must be between 10 and 1000 W/m²-K."),
                'cfd_mesh_size': get_valid_float(self.cfd_mesh_entry, "CFD Mesh Size", 0.01, 5, "CFD Mesh Size must be between 0.01 and 5 mm."),
                'inlet_pressure': get_valid_float(self.pressure_entry, "Inlet Pressure", 50, 300, "Inlet Pressure must be between 50 and 300 bar."),
                'outlet_pressure': get_valid_float(self.outlet_pressure_entry, "Outlet Pressure", 0.1, 10, "Outlet Pressure must be between 0.1 and 10 bar."),
                'turbulence_model': self.turbulence_model.get(),
                'reverse_flow': self.reverse_flow_var.get(),
                'cfd_iterations': get_valid_int(self.cfd_iterations_entry, "CFD Iterations", 0, 100, "CFD Iterations must be between 0 and 100.")
            }

            # Initialize simulation environment
            self.initialize_simulation()
            self.import_geometry()

            results = {}
            # Run selected simulations
            if self.modal_var.get():
                modal_results = self.setup_modal_analysis(params)
                results.update(modal_results)
                logger.info(f"Modal analysis completed. Frequencies: {modal_results['frequencies']}")

            if self.transient_var.get():
                transient_results = self.setup_transient_analysis(params)
                results.update(transient_results)
                logger.info("Transient structural analysis completed.")

            if self.thermal_var.get():
                thermal_results = self.setup_thermal_analysis(params)
                results.update(thermal_results)
                logger.info("Thermal analysis completed.")

            if self.cfd_var.get():
                cfd_results = self.setup_cfd_analysis(params)
                results.update(cfd_results)
                logger.info("CFD analysis completed.")

            # Save results
            result_file = self.save_results(results)
            self.integrate_with_simulink(result_file)
            messagebox.showinfo("Success", f"Simulations completed. Results saved to {result_file}")

        except Exception as e:
            logger.error(f"Simulation failed: {str(e)}", exc_info=True)
            messagebox.showerror("Error", f"Simulation failed: {str(e)}")
        finally:
            if self.eng:
                logger.info("Closing MATLAB engine")
                self.eng.quit()
                self.eng = None

    def save_results(self, results):
        """Save simulation results to output directory on Desktop."""
        try:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = os.path.join(self.output_dir, f"results_{timestamp}.mat")
            scipy.io.savemat(result_file, results)
            logger.info(f"Results saved to {result_file}.")
            return result_file
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}", exc_info=True)
            raise

    def integrate_with_simulink(self, result_file):
        """Integrate results with MATLAB/Simulink."""
        try:
            self.eng = matlab.engine.start_matlab()
            self.eng.load_system('Helicopter_Fuel_Injection', nargout=0)
            self.eng.workspace['sim_results'] = scipy.io.loadmat(result_file)
            self.eng.sim('Helicopter_Fuel_Injection', nargout=0)
            failure_modes = self.eng.workspace['failure_modes'] if 'failure_modes' in self.eng.workspace else None
            logger.info(f"Simulink integration completed. Detected failure modes: {failure_modes}")
        except Exception as e:
            logger.error(f"Error integrating with Simulink: {str(e)}", exc_info=True)
            raise
        finally:
            if self.eng:
                self.eng.quit()
                self.eng = None

def main():
    """Main function to launch the GUI."""
    try:
        root = TkinterDnD.Tk()
        app = DigitalTwinGUI(root)
        root.geometry("600x750")
        root.mainloop()
    except Exception as e:
        logger.error(f"Program failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()