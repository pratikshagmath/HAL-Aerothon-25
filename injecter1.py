import os
import json
import time
import random
import threading
import logging
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
import matlab.engine
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Configure logging
log_file = f"simulation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Simulation parameters
SIMULATION_RATE = 0.1  # seconds between updates
SIMULATION_START_TIME = time.time()
FUEL_TANK_CAPACITY = 500.0  # kg
NOMINAL_RPM = 38000  # Nominal RPM for engine

matlab.double=0.56;

# Shared state for continuous simulation
class SimulationState:
    def __init__(self):
        self.throttle = 50.0  # Default throttle (0-100)
        self.fuel_level = 100.0  # % (full tank initially)
        self.outputs = {
            "altitude": 0.0,  # meters
            "fuelFlow": 0.0305,  # kg/s 
            "pressure": 5.86e6,  # Pa 
            "temperature": 75.0,  # °C
            "efficiency": 94.2,  # %
            "clogRate": 0.0,  # %/s
            "cokeRate": 0.0,  # %/s
            "cokeValue": 0.0,  # %
            "clogLevel": 0.0,  # %
            "thermalRisk": 0.0,  # dimensionless (0-1)
            "sim_time": 0.0,  # seconds
            "rpm": NOMINAL_RPM,  # revolutions per minute
            "deltaP": 0.0,  # Pa
            "isSurging": False,
            "T_inj": 75.0,  # °C
            "AFR_error": 0.0,  # dimensionless
            "fuelPenalty": 0.0,  # dimensionless
            "drift": 0.0,  # dimensionless
            "correctedFuelDelay": 0.0,  # seconds
            "pumpEfficiency": 95.0,  # %
            "tankPressure": 5.86e6,  # Pa
            "fuelLevel": 100.0  # %
        }
        self.running = False

# Physics model for fuel system simulation
class PhysicsModel:
    def __init__(self):
        self.last_update = time.time()
        self.altitude = 0.0  # meters
        self.fuel_flow = 0.0305  # kg/s
        self.pressure = 5.86e6  # Pa
        self.temperature = 75.0  # °C
        self.efficiency = 94.2  # %
        self.clog_rate = 0.0  # %/s
        self.coke_rate = 0.0  # %/s
        self.coke_value = 0.0  # %
        self.clog_level = 0.0  # %
        self.thermal_risk = 0.0  # dimensionless
        self.rpm = NOMINAL_RPM  # rpm
        self.delta_p = 0.0  # Pa
        self.is_surging = False
        self.T_inj = 75.0  # °C
        self.AFR_error = 0.0  # dimensionless
        self.fuel_penalty = 0.0  # dimensionless
        self.drift = 0.0  # dimensionless
        self.corrected_fuel_delay = 0.0  # seconds
        self.pump_efficiency = 95.0  # %
        self.tank_pressure = 5.86e6  # Pa
        self.fuel_level = FUEL_TANK_CAPACITY  # kg 
        
    def update(self, throttle, dt):
        """Update the physics model based on throttle input and time delta"""
        throttle_norm = throttle / 100.0
        
        # Altitude control: simpler model with increased sensitivity
        target_altitude = 25.0  # meters
        if self.altitude < target_altitude:
            altitude_adjustment = (target_altitude - self.altitude) * 0.3
            self.altitude += (throttle_norm * 0.762 - 0.1524 + altitude_adjustment) * dt * 10
        else:
            self.altitude += (throttle_norm * 0.762 - 0.1524) * dt * 10
        self.altitude = max(0, self.altitude)
        
        # RPM: realistic, affected by throttle and failures
        self.rpm = NOMINAL_RPM * (0.8 + 0.4 * throttle_norm) * (1 - self.clog_level / 200 - self.coke_value / 200)
        self.rpm = max(6000, min(15000, self.rpm))
        
        # Fuel flow: based on power demand
        base_fuel_flow = 0.024943 + throttle_norm * 0.012597
        self.fuel_flow = base_fuel_flow * (1 + self.fuel_penalty) / (self.pump_efficiency / 100) + random.uniform(-0.00025, 0.00025)
        self.fuel_flow = max(0.01, min(0.05, self.fuel_flow))
        
        # Fuel level
        self.fuel_level = max(0, self.fuel_level - self.fuel_flow * dt * (1 + 0.1 if self.is_surging else 1))
        fuel_level_percent = (self.fuel_level / FUEL_TANK_CAPACITY) * 100
        
        # Pressure
        self.pressure = 5.52e6 + throttle_norm * 6.89e5 * (self.pump_efficiency / 100) + random.uniform(-3.45e4, 3.45e4)
        self.tank_pressure = max(4.0e6, 5.86e6 - (0.1e6 if self.is_surging else 0) - (0.05e6 if self.fuel_flow < 0.02 else 0))
        
        # Temperature: heat from fuel flow, cooling from RPM
        heat_input = self.fuel_flow * 1000
        cooling = self.rpm / NOMINAL_RPM * 10
        self.temperature = 70.0 + (heat_input - cooling) * 0.1 + random.uniform(-2, 2)
        self.temperature = max(60, min(110, self.temperature))
        
        # Efficiency
        base_efficiency = 92 + (1 - abs(throttle_norm - 0.7)) * 8
        self.efficiency = base_efficiency * (1 - self.clog_level / 200 - self.coke_value / 200) + random.uniform(-0.5, 0.5)
        self.efficiency = max(80, min(99, self.efficiency))
        
        # Failure modes
        self.simulate_failure_modes(throttle_norm, dt)
        self.thermal_risk = min(1.0, max(0.0, 
            (self.temperature - 80) / 15 + 
            (self.pressure - 6.21e6) / 0.69e6 + 
            throttle_norm * 1 + 
            random.uniform(-0.5, 0.5)
        ))
        
        # Pump efficiency
        self.pump_efficiency = max(50, 95 - (self.clog_level / 5) - (self.temperature - 80) / 10)
        
        return {
            "altitude": self.altitude,
            "fuelFlow": self.fuel_flow,
            "pressure": self.pressure,
            "temperature": self.temperature,
            "efficiency": self.efficiency,
            "clogRate": self.clog_rate,
            "cokeRate": self.coke_rate,
            "cokeValue": self.coke_value,
            "clogLevel": self.clog_level,
            "thermalRisk": self.thermal_risk,
            "sim_time": time.time() - SIMULATION_START_TIME,
            "rpm": self.rpm,
            "deltaP": self.delta_p,
            "isSurging": self.is_surging,
            "T_inj": self.T_inj,
            "AFR_error": self.AFR_error,
            "fuelPenalty": self.fuel_penalty,
            "drift": self.drift,
            "correctedFuelDelay": self.corrected_fuel_delay,
            "pumpEfficiency": self.pump_efficiency,
            "tankPressure": self.tank_pressure,
            "fuelLevel": fuel_level_percent
        }
    
    def simulate_failure_modes(self, throttle_norm, dt):
        """Simulate various failure modes based on operating conditions"""
        self.clog_rate = 0.001 + (throttle_norm - 0.5) ** 2 * 0.005 * (1 + self.temperature / 100)
        self.clog_level += self.clog_rate * dt * 10
        self.clog_level = min(100, max(0, self.clog_level))
        self.coke_rate = 0.0005 + (self.temperature - 70) / 1000 * 0.005 * (1 + throttle_norm)
        self.coke_value += self.coke_rate * dt * 10
        self.coke_value = min(100, max(0, self.coke_value))
        self.delta_p = random.uniform(-6.89e4, 6.89e4) + throttle_norm * 3.45e4
        self.is_surging = (self.temperature > 85 and random.random() < 0.05) or \
                         (self.clog_level > 30 and random.random() < 0.1)
        self.T_inj = self.temperature + 5 + random.uniform(-2, 2)
        self.AFR_error = random.uniform(-0.5, 0.5) + (throttle_norm - 0.6) * 0.3
        self.fuel_penalty = max(0, (self.clog_level - 20) / 80 + self.coke_value / 100)
        self.drift = random.uniform(-0.1, 0.1) + (self.rpm - NOMINAL_RPM) / 50000
        self.corrected_fuel_delay = abs(self.drift) * 1.0

# GUI Application
class HelicopterFuelDigitalTwinGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Helicopter Fuel System Digital Twin")
        self.root.geometry("1280x720")
        self.root.configure(bg="#0a0a0a")

        # Simulation state and model
        self.state = SimulationState()
        self.physics_model = PhysicsModel()
        self.state.running = True
        self.historical_data = []
        self.alerts = []
        self.system_status = "OPTIMAL"

        # Create GUI elements
        self.create_gui()

        # Start simulation thread
        self.simulation_thread = threading.Thread(target=self.run_simulation, daemon=True)
        self.simulation_thread.start()

        # Start GUI update loop
        self.update_gui()

    def create_gui(self):
        # Main frame with scrollbar
        self.canvas = tk.Canvas(self.root, bg="#0a0a0a")
        self.scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Header
        tk.Label(self.scrollable_frame, text="Helicopter Fuel System Digital Twin", 
                 font=("Arial", 14, "bold"), bg="#0a0a0a", fg="#ffffff").pack(pady=8)

        # Status frame
        status_frame = tk.Frame(self.scrollable_frame, bg="#0a0a0a")
        status_frame.pack(fill=tk.X, pady=5)
        self.status_label = tk.Label(status_frame, text="System Status: OPTIMAL", 
                                     font=("Arial", 12, "bold"), bg="#0a0a0a", fg="#34c759")
        self.status_label.pack(side=tk.LEFT, padx=10)

        # Throttle control
        throttle_frame = tk.Frame(self.scrollable_frame, bg="#0a0a0a")
        throttle_frame.pack(fill=tk.X, pady=5)
        tk.Label(throttle_frame, text="Throttle: ", font=("Arial", 12), 
                 bg="#0a0a0a", fg="#ffffff").pack(side=tk.LEFT)
        self.throttle_label = tk.Label(throttle_frame, text="50%", font=("Arial", 12), 
                                       bg="#0a0a0a", fg="#ffffff")
        self.throttle_label.pack(side=tk.LEFT, padx=10)
        self.throttle_scale = tk.Scale(throttle_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
                                       length=250, bg="#1f1f1f", fg="#ffffff", 
                                       highlightbackground="#0a0a0a", troughcolor="#4b5563", 
                                       command=self.update_throttle)
        self.throttle_scale.set(50)
        self.throttle_scale.pack(side=tk.LEFT)

        # Metrics frame with individual boxes
        metrics_frame = tk.Frame(self.scrollable_frame, bg="#0a0a0a")
        metrics_frame.pack(fill=tk.X, pady=8)

        # Altitude
        self.altitude_frame = tk.Frame(metrics_frame, bg="#1f1f1f", bd=2, relief="groove")
        self.altitude_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)
        self.altitude_label = tk.Label(self.altitude_frame, text="Altitude: 0.0 m", font=("Arial", 10), 
                                       bg="#1f1f1f", fg="#ffffff", width=15, anchor="w")
        self.altitude_label.pack(padx=5, pady=5)

        # Fuel Flow, Pump Efficiency, Fuel Level, and Gauge
        self.fuel_frame = tk.Frame(metrics_frame, bg="#1f1f1f", bd=2, relief="groove")
        self.fuel_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)
        self.fuel_flow_label = tk.Label(self.fuel_frame, text="Fuel Flow: 0.0305 kg/s", font=("Arial", 10), 
                                        bg="#1f1f1f", fg="#ffffff", width=15, anchor="w")
        self.fuel_flow_label.pack(padx=5, pady=2)
        self.pump_efficiency_label = tk.Label(self.fuel_frame, text="Pump Eff.: 95.0 %", font=("Arial", 10), 
                                             bg="#1f1f1f", fg="#fef08a", width=15, anchor="w")
        self.pump_efficiency_label.pack(padx=5, pady=2)
        self.fuel_level_label = tk.Label(self.fuel_frame, text="Fuel Level: 100.0 %", font=("Arial", 10), 
                                         bg="#1f1f1f", fg="#ffffff", width=15, anchor="w")
        self.fuel_level_label.pack(padx=5, pady=2)
        self.fuel_gauge = ttk.Progressbar(self.fuel_frame, length=100, mode="determinate", maximum=100)
        self.fuel_gauge.pack(padx=5, pady=2)

        # Temperature and Injector Temperature
        self.temp_frame = tk.Frame(metrics_frame, bg="#1f1f1f", bd=2, relief="groove")
        self.temp_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)
        self.temperature_label = tk.Label(self.temp_frame, text="Temperature: 75.0 °C", font=("Arial", 10), 
                                         bg="#1f1f1f", fg="#ffffff", width=15, anchor="w")
        self.temperature_label.pack(padx=5, pady=2)
        self.t_inj_label = tk.Label(self.temp_frame, text="Inj. Temp: 75.0 °C", font=("Arial", 10), 
                                    bg="#1f1f1f", fg="#fef08a", width=15, anchor="w")
        self.t_inj_label.pack(padx=5, pady=2)

        # Pressure, Tank Pressure, and Delta Pressure
        self.pressure_frame = tk.Frame(metrics_frame, bg="#1f1f1f", bd=2, relief="groove")
        self.pressure_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)
        self.pressure_label = tk.Label(self.pressure_frame, text="Pressure: 5.86 MPa", font=("Arial", 10), 
                                       bg="#1f1f1f", fg="#ffffff", width=15, anchor="w")
        self.pressure_label.pack(padx=5, pady=2)
        self.tank_pressure_label = tk.Label(self.pressure_frame, text="Tank Press.: 5.86 MPa", font=("Arial", 10), 
                                           bg="#1f1f1f", fg="#fef08a", width=15, anchor="w")
        self.tank_pressure_label.pack(padx=5, pady=2)
        self.delta_p_label = tk.Label(self.pressure_frame, text="Delta P: 0.0 kPa", font=("Arial", 10), 
                                      bg="#1f1f1f", fg="#fef08a", width=15, anchor="w")
        self.delta_p_label.pack(padx=5, pady=2)

        # RPM, Efficiency, and AFR Error
        self.other_frame = tk.Frame(metrics_frame, bg="#1f1f1f", bd=2, relief="groove")
        self.other_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)
        self.rpm_label = tk.Label(self.other_frame, text="RPM: 38000", font=("Arial", 10), 
                                  bg="#1f1f1f", fg="#ffffff", width=15, anchor="w")
        self.rpm_label.pack(padx=5, pady=2)
        self.efficiency_label = tk.Label(self.other_frame, text="Efficiency: 94.2 %", font=("Arial", 10), 
                                         bg="#1f1f1f", fg="#ffffff", width=15, anchor="w")
        self.efficiency_label.pack(padx=5, pady=2)
        self.afr_error_label = tk.Label(self.other_frame, text="AFR Error: 0.0", font=("Arial", 10), 
                                        bg="#1f1f1f", fg="#fef08a", width=15, anchor="w")
        self.afr_error_label.pack(padx=5, pady=2)

        # Failure modes frame with individual boxes
        failure_frame = tk.Frame(self.scrollable_frame, bg="#0a0a0a")
        failure_frame.pack(fill=tk.X, pady=8)
        tk.Label(failure_frame, text="Failure Modes", font=("Arial", 12, "bold"), 
                 bg="#0a0a0a", fg="#ffffff").pack(anchor="w")

        # Thermal Risk
        self.thermal_risk_frame = tk.Frame(failure_frame, bg="#1f1f1f", bd=2, relief="groove")
        self.thermal_risk_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)
        self.thermal_risk_label = tk.Label(self.thermal_risk_frame, text="Thermal Risk: 0 %", font=("Arial", 10), 
                                           bg="#1f1f1f", fg="#34c759", width=15, anchor="w")
        self.thermal_risk_label.pack(padx=5, pady=5)

        # Clog Rate
        self.clog_rate_frame = tk.Frame(failure_frame, bg="#1f1f1f", bd=2, relief="groove")
        self.clog_rate_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)
        self.clog_rate_label = tk.Label(self.clog_rate_frame, text="Clog Rate: 0.000 %/s", font=("Arial", 10), 
                                        bg="#1f1f1f", fg="#34c759", width=15, anchor="w")
        self.clog_rate_label.pack(padx=5, pady=5)

        # Clog Level
        self.clog_level_frame = tk.Frame(failure_frame, bg="#1f1f1f", bd=2, relief="groove")
        self.clog_level_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)
        self.clog_level_label = tk.Label(self.clog_level_frame, text="Clog Level: 0.0 %", font=("Arial", 10), 
                                         bg="#1f1f1f", fg="#34c759", width=15, anchor="w")
        self.clog_level_label.pack(padx=5, pady=5)

        # Coke Rate
        self.coke_rate_frame = tk.Frame(failure_frame, bg="#1f1f1f", bd=2, relief="groove")
        self.coke_rate_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)
        self.coke_rate_label = tk.Label(self.coke_rate_frame, text="Coke Rate: 0.000 %/s", font=("Arial", 10), 
                                        bg="#1f1f1f", fg="#34c759", width=15, anchor="w")
        self.coke_rate_label.pack(padx=5, pady=5)

        # Coke Value
        self.coke_value_frame = tk.Frame(failure_frame, bg="#1f1f1f", bd=2, relief="groove")
        self.coke_value_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)
        self.coke_value_label = tk.Label(self.coke_value_frame, text="Coke Value: 0.0 %", font=("Arial", 10), 
                                         bg="#1f1f1f", fg="#34c759", width=15, anchor="w")
        self.coke_value_label.pack(padx=5, pady=5)

        # Plot frame for side-by-side graphs
        plot_frame = tk.Frame(self.scrollable_frame, bg="#0a0a0a")
        plot_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Altitude plot
        altitude_plot_frame = tk.Frame(plot_frame, bg="#0a0a0a")
        altitude_plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.altitude_fig = Figure(figsize=(5, 2), dpi=100)
        self.altitude_ax = self.altitude_fig.add_subplot(111)
        self.altitude_ax.set_xlabel("Time")
        self.altitude_ax.set_ylabel("Altitude (m)")
        self.altitude_ax.grid(True, linestyle="--", alpha=0.7, color="#4b5563")
        self.altitude_canvas = FigureCanvasTkAgg(self.altitude_fig, master=altitude_plot_frame)
        self.altitude_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Fuel flow plot
        fuel_plot_frame = tk.Frame(plot_frame, bg="#0a0a0a")
        fuel_plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.fuel_fig = Figure(figsize=(5, 2), dpi=100)
        self.fuel_ax = self.fuel_fig.add_subplot(111)
        self.fuel_ax.set_xlabel("Time")
        self.fuel_ax.set_ylabel("Fuel Flow (kg/s)")
        self.fuel_ax.grid(True, linestyle="--", alpha=0.7, color="#4b5563")
        self.fuel_canvas = FigureCanvasTkAgg(self.fuel_fig, master=fuel_plot_frame)
        self.fuel_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Alerts frame
        alerts_frame = tk.Frame(self.scrollable_frame, bg="#0a0a0a")
        alerts_frame.pack(fill=tk.X, pady=5)
        tk.Label(alerts_frame, text="Alerts & Preventive Measures", font=("Arial", 12, "bold"), 
                 bg="#0a0a0a", fg="#ffffff").pack(anchor="w")
        self.alerts_text = tk.Text(alerts_frame, height=5, bg="#1f1f1f", fg="#ffffff", 
                                   font=("Arial", 10), wrap=tk.WORD)
        self.alerts_text.pack(fill=tk.X, pady=5)

    def update_throttle(self, value):
        self.state.throttle = float(value)
        self.throttle_label.config(text=f"{float(value):.0f}%")

    def run_simulation(self):
        last_time = time.time()
        while self.state.running:
            try:
                current_time = time.time()
                dt = current_time - last_time
                last_time = current_time
                self.state.outputs = self.physics_model.update(self.state.throttle, dt)
                time.sleep(max(0, SIMULATION_RATE - (time.time() - current_time)))
            except Exception as e:
                logger.error(f"Simulation crashed: {str(e)}", exc_info=True)
                self.state.running = False
                self.root.after(0, lambda: messagebox.showerror("Simulation Crash", f"The simulation has crashed: {str(e)}"))
                break

    def update_gui(self):
        if self.state.running:
            data = self.state.outputs
            # Update metrics
            self.altitude_label.config(text=f"Altitude: {data['altitude']:.1f} m")
            self.fuel_flow_label.config(text=f"Fuel Flow: {data['fuelFlow']:.5f} kg/s")
            self.pump_efficiency_label.config(text=f"Pump Eff.: {data['pumpEfficiency']:.1f} %", 
                                             fg="#f97316" if data['pumpEfficiency'] < 80 else "#fef08a")
            self.fuel_level_label.config(text=f"Fuel Level: {data['fuelLevel']:.1f} %",
                                         fg="#dc2626" if data['fuelLevel'] < 10 else "#f97316" if data['fuelLevel'] < 20 else "#ffffff")
            self.fuel_gauge['value'] = data['fuelLevel']
            self.temperature_label.config(text=f"Temperature: {data['temperature']:.1f} °C")
            self.t_inj_label.config(text=f"Inj. Temp: {data['T_inj']:.1f} °C",
                                    fg="#f97316" if data['T_inj'] > 90 else "#fef08a")
            self.pressure_label.config(text=f"Pressure: {data['pressure']/1e6:.2f} MPa")
            self.tank_pressure_label.config(text=f"Tank Press.: {data['tankPressure']/1e6:.2f} MPa", 
                                           fg="#f97316" if data['tankPressure'] < 4.5e6 else "#fef08a")
            self.delta_p_label.config(text=f"Delta P: {data['deltaP']/1e3:.1f} kPa",
                                      fg="#f97316" if abs(data['deltaP']) > 5e4 else "#fef08a")
            self.rpm_label.config(text=f"RPM: {data['rpm']:.0f}",
                                  fg="#f97316" if data['rpm'] < 8000 or data['rpm'] > 15000 else "#ffffff")
            self.efficiency_label.config(text=f"Efficiency: {data['efficiency']:.1f} %",
                                         fg="#f97316" if data['efficiency'] < 85 else "#ffffff")
            self.afr_error_label.config(text=f"AFR Error: {data['AFR_error']:.2f}",
                                        fg="#f97316" if abs(data['AFR_error']) > 0.4 else "#fef08a")

            # Update failure modes
            self.thermal_risk_label.config(text=f"Thermal Risk: {(data['thermalRisk'] * 100):.0f} %",
                                          fg="#dc2626" if data['thermalRisk'] > 0.8 else "#f97316" if data['thermalRisk'] > 0.5 else "#34c759")
            self.clog_rate_label.config(text=f"Clog Rate: {data['clogRate']:.3f} %/s",
                                        fg="#f97316" if data['clogRate'] > 0.005 else "#34c759")
            self.clog_level_label.config(text=f"Clog Level: {data['clogLevel']:.1f} %",
                                         fg="#dc2626" if data['clogLevel'] > 30 else "#34c759")
            self.coke_rate_label.config(text=f"Coke Rate: {data['cokeRate']:.3f} %/s",
                                        fg="#f97316" if data['cokeRate'] > 0.003 else "#34c759")
            self.coke_value_label.config(text=f"Coke Value: {data['cokeValue']:.1f} %",
                                         fg="#dc2626" if data['cokeValue'] > 20 else "#34c759")

            # Update historical data
            time_str = datetime.now().strftime("%H:%M:%S")
            self.historical_data.append({
                "time": time_str,
                "altitude": data["altitude"],
                "fuelFlow": data["fuelFlow"],
                "pressure": data["pressure"],
                "temperature": data["temperature"],
                "efficiency": data["efficiency"],
            })
            self.historical_data = self.historical_data[-100:]

            # Update system status
            if data["thermalRisk"] > 0.8 or data["isSurging"] or data["cokeValue"] > 20 or data["fuelLevel"] < 10 or data["rpm"] < 8000 or data["rpm"] > 15000:
                self.system_status = "WARNING"
                self.status_label.config(text="System Status: WARNING", fg="#dc2626")
            elif data["clogLevel"] > 30 or data["thermalRisk"] > 0.5 or data["pumpEfficiency"] < 80 or data["tankPressure"] < 4.5e6 or data["fuelLevel"] < 20:
                self.system_status = "CAUTION"
                self.status_label.config(text="System Status: CAUTION", fg="#f97316")
            else:
                self.system_status = "OPTIMAL"
                self.status_label.config(text="System Status: OPTIMAL", fg="#34c759")

            # Update alerts with preventive measures
            new_alerts = []
            if data["thermalRisk"] > 0.8:
                new_alerts.append({
                    "type": "danger",
                    "message": f"CRITICAL THERMAL RISK: {(data['thermalRisk'] * 100):.0f}% - Reduce throttle or increase cooling.",
                    "time": time_str
                })
            if data["isSurging"]:
                new_alerts.append({
                    "type": "warning",
                    "message": "COMBUSTION SURGE DETECTED - Adjust fuel flow or check injector.",
                    "time": time_str
                })
            if data["clogLevel"] > 30:
                new_alerts.append({
                    "type": "warning",
                    "message": f"INJECTOR CLOGGING: {data['clogLevel']:.1f}% - Schedule injector cleaning.",
                    "time": time_str
                })
            if data["cokeValue"] > 20:
                new_alerts.append({
                    "type": "warning",
                    "message": f"HIGH COKE DEPOSITION: {data['cokeValue']:.1f}% - Use fuel additives or inspect system.",
                    "time": time_str
                })
            if data["pumpEfficiency"] < 80:
                new_alerts.append({
                    "type": "warning",
                    "message": f"LOW PUMP EFFICIENCY: {data['pumpEfficiency']:.1f}% - Check pump for wear or blockages.",
                    "time": time_str
                })
            if data["tankPressure"] < 4.5e6:
                new_alerts.append({
                    "type": "warning",
                    "message": f"LOW TANK PRESSURE: {data['tankPressure']/1e6:.2f} MPa - Inspect tank seals or pressurization system.",
                    "time": time_str
                })
            if data["fuelLevel"] < 20:
                new_alerts.append({
                    "type": "warning",
                    "message": f"LOW FUEL LEVEL: {data['fuelLevel']:.1f}% - Plan for refueling.",
                    "time": time_str
                })
            if data["fuelLevel"] < 10:
                new_alerts.append({
                    "type": "danger",
                    "message": f"CRITICAL FUEL LEVEL: {data['fuelLevel']:.1f}% - Immediate landing required.",
                    "time": time_str
                })
            if data["rpm"] < 8000:
                new_alerts.append({
                    "type": "warning",
                    "message": f"LOW RPM: {data['rpm']:.0f} - Check engine load or fuel system.",
                    "time": time_str
                })
            if data["rpm"] > 15000:
                new_alerts.append({
                    "type": "warning",
                    "message": f"HIGH RPM: {data['rpm']:.0f} - Reduce throttle or inspect governor.",
                    "time": time_str
                })
            self.alerts = new_alerts + self.alerts[:9]
            self.alerts_text.delete(1.0, tk.END)
            for alert in self.alerts:
                color = "#dc2626" if alert["type"] == "danger" else "#f97316" if alert["type"] == "warning" else "#ffffff"
                self.alerts_text.insert(tk.END, f"{alert['time']}: {alert['message']}\n", color)
                self.alerts_text.tag_configure(color, foreground=color)

            # Update altitude plot
            self.altitude_ax.clear()
            times = [d["time"] for d in self.historical_data]
            altitudes = [d["altitude"] for d in self.historical_data]
            self.altitude_ax.plot(times, altitudes, label="Altitude (m)", color="#8884d8")
            self.altitude_ax.set_xlabel("Time")
            self.altitude_ax.set_ylabel("Altitude (m)")
            self.altitude_ax.legend()
            self.altitude_ax.grid(True, linestyle="--", alpha=0.7, color="#4b5563")
            self.altitude_ax.tick_params(axis="x", rotation=45)
            self.altitude_canvas.draw()

            # Update fuel flow plot
            self.fuel_ax.clear()
            fuel_flows = [d["fuelFlow"] for d in self.historical_data]
            self.fuel_ax.plot(times, fuel_flows, label="Fuel Flow (kg/s)", color="#82ca9d")
            self.fuel_ax.set_xlabel("Time")
            self.fuel_ax.set_ylabel("Fuel Flow (kg/s)")
            self.fuel_ax.legend()
            self.fuel_ax.grid(True, linestyle="--", alpha=0.7, color="#4b5563")
            self.fuel_ax.tick_params(axis="x", rotation=45)
            self.fuel_canvas.draw()

        self.root.after(int(SIMULATION_RATE * 1000), self.update_gui)

    def on_closing(self):
        self.state.running = False
        logger.info("GUI closed by user.")
        self.root.destroy()

def main():
    try:
        root = tk.Tk()
        app = HelicopterFuelDigitalTwinGUI(root)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        root.mainloop()
    except Exception as e:
        logger.error(f"Program failed: {str(e)}", exc_info=True)
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Program Crash", f"The program has crashed: {str(e)}")
        root.destroy()
        raise

if __name__ == "__main__":
    main()