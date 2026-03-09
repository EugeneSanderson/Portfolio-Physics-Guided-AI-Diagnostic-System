"""
Distillation Column Digital Twin & Fault Diagnostic Suite
Author: Eugene Sanderson
"""

# =====================================
# Imports
# =====================================

import Scripts.Dataset_Generator as Dataset_Generator

import Scripts.Column_RF_Trainer
import Scripts.Column_Generate_Matrices

import Scripts.Physics_State_Inference_Trainer_Compositions
import Scripts.Physics_State_Inference_Trainer_Flow_Rates
import Scripts.Physics_State_Inference_Trainer_Steam_Reflux_Pressure
import Scripts.Physics_State_Inference_Trainer_Temperature

import Scripts.Soft_Sensor_Composition
import Scripts.Soft_Sensor_Flows
import Scripts.Soft_Sensor_SRP
import Scripts.Soft_Sensor_Temperatures

import Scripts.Append_Residuals

import Scripts.True_State_Trainer
import Scripts.True_State_Comparer

import Scripts.NN_Trainer_Fault_Tray_Locator
import Scripts.NN_Fault_Detector

import Scripts.Print_Diagnostics

# =====================================
# Plant Data - CSV format
# =====================================

plant_data = "Data/Plant_Data_Example.csv"           #This is your plant data
dataset_csv = "Data/column_diagnostic_dataset.csv"   #The CSV the simulation model will use

# =====================================
# Parameters
# =====================================
# --=== Column Targets ===--

column_params ={"Xf" : 1.0,                     #Feed flow rate
                "X1f" : 0.5,                    #Feed EtOH Composition
                "pf" : 81.325,                  #Column operating pressure
                "Tf" : 82.75,                   #Feed temperature
                "Xd" : 0.6,                     #Distillate flow rate
                "X1d" : 0.8,                    #Distillate EtOH composition
                "Xb" : 0.4,                     #Bottom flow rate
                "X1b" : 0.005,                  #Bottom EtOH composition
                "R" : 1.5,                      #Reflux ratio
                "steam_factor" : 1.0,           #Steam ratio
                "RectifyingTrays_Setup" : [],   #Dynamically set up, do not change
                "StrippingTrays_Setup" : [],    #Dynamically set up, do not change
                }

# --=== Dataset Generation ===--

dataset_ranges = {"R_MIN" : 1.125,              #Minimum reflux ratio in dataset
                  "R_MAX" : 3.0,                #Maximum reflux ratio in dataset
                  "STEAM_MIN" : 0.9,            #Minimum steam ratio in dataset
                  "STEAM_MAX" : 2.0,            #Maximum steam ratio in dataset
                  "Process_Deviation" : 0.05    #Adds 5% variation to column_params
}

N_SAMPLES = 50000                                #How many simulations to append to the dataset

# --=== Soft Sensors Neural Network Configs ===--

psitc_params = {"hidden_layer_sizes" : (128,64,32),                 # Composition Soft Sensors (Full Profile) Neural Network
                "activation":"relu",
                "max_iter":5000,
                "learning_rate":"adaptive",
                "verbose":True,
                "tol":0.000001}
psitfr_params = {"hidden_layer_sizes" : (128,64,32),                # Feed Rates Soft Sensors (Feed, Distillate, Bottoms) Neural Network
                "activation":"relu",
                "max_iter":5000,
                "learning_rate":"adaptive",
                "verbose":True,
                "tol":0.000001}
psitsrp_params = {"hidden_layer_sizes" : (128,64,32),               # Steam, Reflux, Column Pressure Soft Sensors Neural Network
                    "activation":"relu",
                    "max_iter":5000,
                    "learning_rate":"adaptive",
                    "verbose":True,
                    "tol":0.000001}
psitt_params = {"hidden_layer_sizes" : (128,64,32),                 # Temperature Soft Sensors (Full Profile) Neural Network
                "activation":"relu",
                "max_iter":5000,
                "learning_rate":"adaptive",
                "verbose":True,
                "tol":0.000001}

# --=== True State Neural Network Config ===--

tst_params = {"hidden_layer_sizes" : (128,64,32),                   # True State Profiler Neural Network
                "activation":"relu",
                "max_iter":5000,
                "learning_rate":"adaptive",
                "verbose":True,
                "tol":0.000001}

# --=== Fault Model Neural Network Config ===--

nntftl_params = {"hidden_layer_sizes" : (128,64,32),                # Fault detector Neural Network
                "activation":"relu",
                "max_iter":5000,
                "learning_rate":"adaptive",
                "verbose":True,
                "tol":0.000001}
# =====================================
# Pipeline steps
# =====================================

def generate_dataset():
    print("\n=== DATASET GENERATION ===")
    Scripts.Dataset_Generator.generate(column_params,dataset_ranges,N_SAMPLES,dataset_csv)

def train_random_forest():
    print("\n=== RANDOM FOREST TRAINING ===")
    Scripts.Column_RF_Trainer.train(dataset_csv)
    Scripts.Column_Generate_Matrices.generate_graphs(dataset_csv)

def train_soft_sensors():
    print("\n=== SOFT SENSOR TRAINING ===")

    psitc_mae = Scripts.Physics_State_Inference_Trainer_Compositions.train(psitc_params,dataset_csv)
    psitfr_mae = Scripts.Physics_State_Inference_Trainer_Flow_Rates.train(psitfr_params,dataset_csv)
    psitsrp_mae = Scripts.Physics_State_Inference_Trainer_Steam_Reflux_Pressure.train(psitsrp_params,dataset_csv)
    psitt_mae = Scripts.Physics_State_Inference_Trainer_Temperature.train(psitt_params,dataset_csv)
    return psitc_mae, psitfr_mae, psitsrp_mae, psitt_mae

def generate_soft_sensor_profiles():
    print("\n=== SOFT SENSOR INFERENCE ===")

    Scripts.Soft_Sensor_Composition.run(dataset_csv)
    Scripts.Soft_Sensor_Flows.run(dataset_csv)
    Scripts.Soft_Sensor_SRP.run(dataset_csv)
    Scripts.Soft_Sensor_Temperatures.run(dataset_csv)

def build_true_state():
    print("\n=== TRUE STATE ESTIMATION ===")

    Scripts.Append_Residuals.run(dataset_csv)
    tst_mae = Scripts.True_State_Trainer.train(tst_params,dataset_csv)
    Scripts.True_State_Comparer.run(dataset_csv)
    return tst_mae

def train_fault_models():
    print("\n=== FAULT DETECTION TRAINING ===")

    nntftl_mae = Scripts.NN_Trainer_Fault_Tray_Locator.train(nntftl_params,dataset_csv)
    return nntftl_mae


# =====================================
# Main pipeline
# =====================================

def main():

    print("\nDISTILLATION DIGITAL TWIN PIPELINE")
    print("===================================")

    generate_dataset()
    train_random_forest()
    psitc_mae, psitfr_mae, psitsrp_mae, psitt_mae = train_soft_sensors()
    generate_soft_sensor_profiles()
    tst_mae = build_true_state()
    nntftl_mae = train_fault_models()

    print("\nPipeline complete.")
    print(f"Soft_Sensors_Compositions Mae: {psitc_mae}")
    print(f"Soft_Sensors_Flow_Rates Mae: {psitfr_mae}")
    print(f"Soft_Sensors_Steam_Reflux_Pressure Mae: {psitsrp_mae}")
    print(f"Soft_Sensors_Temperatures Mae: {psitt_mae}")
    print(f"True_State Mae: {tst_mae}")
    print(f"Fault_Model Mae: {nntftl_mae}")

    return()

def feed_plant_data():

    print("\n=== SOFT SENSOR Prediction ===")
    Scripts.Soft_Sensor_Composition.run(plant_data)
    Scripts.Soft_Sensor_Flows.run(plant_data)
    Scripts.Soft_Sensor_SRP.run(plant_data)
    Scripts.Soft_Sensor_Temperatures.run(plant_data)

    print("\n=== TRUE STATE ESTIMATION ===")
    Scripts.Append_Residuals.run(plant_data)
    Scripts.True_State_Comparer.run(plant_data)

    print("\n=== FAULT DETECTION ===")
    Scripts.NN_Fault_Detector.run(plant_data)

def print_diagnostics():
    Scripts.Print_Diagnostics.print_diagnostics(plant_data)
    Scripts.Print_Diagnostics.plot_diagnostics(plant_data)
# =====================================
# Entry point
# =====================================

if __name__ == "__main__":
    main()                          # Generates the dataset and trains all the models
    feed_plant_data()               # Uses neural nets to implement soft sensors
    print_diagnostics()             # Prints the diagnostics as predicted by the neural networks
