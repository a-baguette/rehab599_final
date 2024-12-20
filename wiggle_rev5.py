'''
Course: REHAB 599 F24 University of Alberta
Author: Annette Lau
Email: annette1@ualberta.ca
Update rev5: changed cumulant to be present position instead

Example of implementing a horde of on-policy General Value Function (GVF) learners.
Connects to 6 Dynamixel XL330-M288-T servo motors in the HANDi Hand to access position
and velocity in real time. The GVF learners predict how much movement each motor will
produce as they open and close one after the other with a fixed gamma value using
the TD(lambda) algorithm.

====================================
1. IMPORTS AND CONSTANTS
====================================
'''
import os, time, numpy as np, csv
from dynamixel_sdk import *
import matplotlib.pyplot as plt
import pandas as pd

# Control Table Addresses for Dynamixel XL330-M288-T Motors
ADDR_PWM_LIMIT = 36
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_PWM = 100
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_PWM = 124
ADDR_PRESENT_CURRENT = 126
ADDR_PRESENT_VELOCITY = 128
ADDR_PRESENT_POSITION = 132

# Communication Settings
COMM_SUCCESS = 0
COMM_TX_FAIL = -1001
PROTOCOL_VERSION = 2.0
BAUDRATE = 1000000 # Common baudrate for communicating with Dynamixels
DEVICENAME = 'COM8'

# Motor Control Values
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
DXL_GOAL_PWM_VALUE = 177 # Change this value to be faster if needed
DXL_PWM_LIMIT_VALUE = 354
DXL_MOVING_STATUS_THRESHOLD = 10

'''
====================================
2. MOTOR CONFIGURATION
====================================
'''
num_fingers = 6
MOTORS = {
    0: {"id": 0, "name": "D0", "min_pos": 1200, "max_pos": 1875},  # Moves opposite, if D0 crashes into D2, changes min_pos to 1500
    1: {"id": 1, "name": "D1", "min_pos": 950, "max_pos": 2048},   # Moves opposite
    2: {"id": 2, "name": "D2", "min_pos": 2048, "max_pos": 4095},
    3: {"id": 3, "name": "D3", "min_pos": 2048, "max_pos": 4095},
    4: {"id": 4, "name": "D4", "min_pos": 2048, "max_pos": 4095},
    5: {"id": 5, "name": "D5", "min_pos": 210, "max_pos": 2048}    # Moves opposite
}

'''
====================================
3. HELPER CLASSES
====================================
'''
# General Value Function express predictions
class GVF:
    def __init__(self, gamma, llama, alpha, state_size):
        self.gamma = gamma # Discount factor (how far in the future predictions span)
        self.llama = llama # Trace decay parameter (how much past affects current state)
        self.alpha = alpha # Learning rate
        self.delta = 0 # TD error
        self.e = np.zeros(state_size) # Eligibility trace vector
        self.w = np.zeros(state_size) # Weight vector
        self.S = np.zeros(state_size) # Current state feature vector x(S)
        self.predict = 0.5
    def update(self, cumulant, S_prime, gamma):
        self.gamma = gamma
        self.predict = np.dot(self.w, self.S) # Prediction for the current state
        self.delta = cumulant + self.gamma*np.dot(self.w, S_prime) - self.predict # TD error update
        self.e = np.minimum(self.e*self.gamma*self.llama + self.S, 1) # Capped eligibility trace update
        self.w = self.w + self.alpha*self.delta*self.e # Weight vector update
        self.S = S_prime # Store current state for the next update

'''
====================================
4. HELPER FUNCTIONS
====================================
'''
# Reads the motor's present position and velocity in real-time
# Outputs an array of very large numbers
# E.g. [pos_D0, vel_D0, pos_D1, vel_D1, ...]
def get_motor_info():
    motor_states = []
    for motor_id in MOTORS.keys():
        dxl_present_position, _, _ = packetHandler.read4ByteTxRx(portHandler, motor_id, ADDR_PRESENT_POSITION)
        dxl_present_velocity, _, _ = packetHandler.read4ByteTxRx(portHandler, motor_id, ADDR_PRESENT_VELOCITY)
        motor_states.extend([dxl_present_position, dxl_present_velocity])
    return np.array(motor_states)

# Takes the position values and normalizes them consistently between 0 (open) and 1 (closed)
# Accounts for whether the motors close at their min or max position
def normalize_position(position, motor_id):
    motor_info = MOTORS[motor_id]
    if motor_id in [0, 1, 5]:  # Motors that close at min_pos. For these motors, min_pos (closed) = 1.0, max_pos (open) = 0.0
        return 1.0 - (position - motor_info["min_pos"]) / (motor_info["max_pos"] - motor_info["min_pos"])
    else:  # Motors that close at max_pos. For these motors, max_pos (closed) = 1.0, min_pos (open) = 0.0
        return (position - motor_info["min_pos"]) / (motor_info["max_pos"] - motor_info["min_pos"])

# Normalizes the large position and velocity values with respect to their byte size
# Outputs an array with values between 0 and 1
# E.g. [pos_D0, vel_D0, pos_D1, vel_D1, ...]
def normalize(raw_state):
    normalized_states = []
    for i, motor_id in enumerate(MOTORS.keys()):
        state_inx = i * 2
        norm_position = normalize_position(raw_state[state_inx], motor_id)
        norm_velocity = (raw_state[state_inx + 1])/(4294967295)
        normalized_states.extend([norm_position, norm_velocity])
    return np.array(normalized_states)

# Makes a 2D array for each motor containing a single active tile
# Flattens all arrays into one large vector containing all 0s and six 1s
def binning(values, num_bins):
    feature_vectors = []
    for i in range(len(MOTORS)):
        base_idx = i * 2
        motor_values = values[base_idx:base_idx + 2]
        array = np.zeros((num_bins, num_bins))
        bin_indices = [min(int(value * num_bins), num_bins - 1) for value in motor_values]
        array[bin_indices[0], bin_indices[1]] = 1
        feature_vectors.extend(array.flatten())
    return np.array(feature_vectors)

# Computes the sum of true returns for the GVF using the past few gamma and cumulant values
# Determined by the size of gamma_list and cumulant_list
def compute_true_return(gamma_list, cumulant_list):
    true_return = 0
    for k, cumulant in enumerate(cumulant_list): # For each k, multiply cumulant by gamma^k
        gamma_product = 1
        for i in range(k): # Calculate γᵏ by multiplying k gammas
            gamma_product *= gamma_list[i]
        true_return += gamma_product * cumulant
    return true_return

# Resets cycle flags when all motors have completed their open/close cycles
def reset_motor_cycles():
    global completed_cycle, started_opening, indices
    completed_cycle = [False] * num_fingers
    started_opening = [False] * num_fingers
    started_opening[0] = True  # Start first motor again
    indices = [0] * num_fingers

'''
====================================
5. MOTOR SETUP AND INITIALIZATION
====================================
'''
# Set up port and packet handlers
portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

# Open port
if portHandler.openPort():
    print("Successfully opened the port!!")
else:
    print("Failed to open the port :(")
    input("Press any key to terminate...")
    quit()

# Set baudrate
if portHandler.setBaudRate(BAUDRATE):
    print("Successfully changed the baudrate!!")
else:
    print("Failed to change the baudrate :(")
    input("Press any key to terminate...")
    quit()

# Enable torque and set PWM for all motors
for motor_id in MOTORS.keys():
    # Enable torque
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, motor_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
    if dxl_comm_result != COMM_SUCCESS:
        print(f"Motor {motor_id}: %s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print(f"Motor {motor_id}: %s" % packetHandler.getRxPacketError(dxl_error))
    else:
        print(f"Motor {motor_id} has been successfully connected")

    # Set PWM (can also set velocity profile if want to control speed of finger open/close)
    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, motor_id, ADDR_GOAL_PWM, DXL_GOAL_PWM_VALUE)
    if dxl_comm_result != COMM_SUCCESS:
        print(f"Motor {motor_id}: %s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print(f"Motor {motor_id}: %s" % packetHandler.getRxPacketError(dxl_error))
    else:
        print(f"Motor {motor_id} PWM limit set successfully")

# Create CSV file and write headers
script_filename = os.path.splitext(os.path.basename(__file__))[0]
csv_filename = f"{script_filename}.csv"
csv_file = open(csv_filename, "w", newline='')
csv_writer = csv.writer(csv_file)

header = ["Time_Step"]
header.extend([f"Pred_{MOTORS[i]['name']}" for i in range(num_fingers)])  # Prediction columns
header.extend([f"Cumu_{MOTORS[i]['name']}" for i in range(num_fingers)])  # Cumulant columns
header.extend([f"TRet_{MOTORS[i]['name']}" for i in range(num_fingers)])  # True Return columns
header.extend([f"Pos_{MOTORS[i]['name']}" for i in range(num_fingers)])   # Present Position columns
for i in range(num_fingers):
    header.extend([f"NormState_{MOTORS[i]['name']}_Pos", 
                  f"NormState_{MOTORS[i]['name']}_Vel"])
csv_writer.writerow(header)

'''
====================================
6. MAIN PROGRAM VARIABLES
====================================
'''
# GVF variables
gamma = 0.9
llama = 0.9
alpha = 0.05
num_bins = 30
states = 2
state_size = num_bins**states * num_fingers

# Counter variables
time_step = 0
stop_time = 2000  # Each cycle of finger movements takes about 50 time steps
window_size = int(1/(1-gamma))

# Initialize TD lambda learners
td = [None]*num_fingers
for i in range(num_fingers):
    td[i] = GVF(gamma, llama, alpha, state_size)

# Initialize histories for calculating true return
cumulant_histories = [[] for _ in range(num_fingers)]
gamma_histories = [[] for _ in range(num_fingers)]
true_returns = [0] * num_fingers

# Initialize goal positions
goal_positions = []
indices = [0] * num_fingers  # Track position index for each motor
for motor_id, motor_info in MOTORS.items():
    if motor_id in [0, 1, 5]:  # Motors that move in opposite direction
        # Start with max_pos (open) for these motors
        goal_positions.append([motor_info["min_pos"], motor_info["max_pos"]])
    else:
        # Start with min_pos (open) for other motors
        goal_positions.append([motor_info["max_pos"], motor_info["min_pos"]])

completed_cycle = [False] * num_fingers  # Track if motor completed open-close cycle
started_opening = [False] * num_fingers  # Track if motor started opening movement

'''
====================================
7. MAIN LOOP
====================================
'''
while time_step <= stop_time:
    current_time = time.time()
    
    # Updates goal positions for each motor
    for i in range(num_fingers):
        motor_id = i
        
        # Check if this motor should start moving
        if not started_opening[i]:
            if i == 0 or (i > 0 and completed_cycle[i-1]):  # Start if first motor or previous completed
                started_opening[i] = True
                print(f"Starting motor {MOTORS[i]['name']}")
        
        # Only move motor if it has been started and not completed cycle
        if started_opening[i] and not completed_cycle[i]:
            # Send goal position to motor
            current_goal = goal_positions[i][indices[i]]
            dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(
                portHandler, motor_id, ADDR_GOAL_POSITION, current_goal)
            
            # Check if motor reached goal position
            dxl_present_position, _, _ = packetHandler.read4ByteTxRx(
                portHandler, motor_id, ADDR_PRESENT_POSITION)
            
            if abs(dxl_present_position - current_goal) < 300:
                if not completed_cycle[i]:
                    indices[i] = (indices[i] + 1) % 2  # Switch to next position
                    if indices[i] == 0:
                        completed_cycle[i] = True
    
    # Get state information and bin it
    raw_state = get_motor_info()
    norm_state = normalize(raw_state)
    motor_state = binning(norm_state, num_bins)

    # Calculates cumulants and update GVFs for each motor
    for i in range(num_fingers):
        motor_id = i
        
        # Sets the normalized present position as the cumulant (1.0 = closed, 0.0 = open)
        # Updates the cumulant and gamma histories
        cumulant = normalize_position(raw_state[i*states], motor_id)
        cumulant_histories[i].append(cumulant)
        gamma_histories[i].append(gamma)

        # Calculate true return for this motor
        if len(cumulant_histories[i]) > window_size:
            true_returns[i] = compute_true_return(gamma_histories[i][-window_size:], cumulant_histories[i][-window_size:])
        
        # Update GVF for this motor
        td[i].update(cumulant, motor_state, gamma)
        
    # Log predictions and true returns for all motors
    norm_predictions = [gvf.predict * (1-gamma) for gvf in td]
    norm_true_returns = [tr * (1-gamma) for tr in true_returns]
    
    # Write to CSV file
    row_data = [time_step]
    row_data.extend(norm_predictions)
    row_data.extend([cumulant_histories[i][-1] for i in range(num_fingers)])
    row_data.extend(norm_true_returns)
    row_data.extend([normalize_position(raw_state[i*states], i) for i in range(num_fingers)])
    row_data.extend(norm_state)
    csv_writer.writerow(row_data)
    
    # For tracking what time step we're on as the program is running
    print(f"Time Step: {time_step}")

    # Uncomment below if predictions, true return, and cumulant need to be tracked
    # print(f"Motor State: {norm_state}")
    # for i in range(num_fingers):
    #     print(f"Motor {MOTORS[i]['name']}: Pred={norm_predictions[i]:.2f}, "
    #           f"Cumu={cumulant_histories[i][-1]:.2f}, "
    #           f"TRet={norm_true_returns[i]:.2f}")

    # Resets the motor's open-close movement cycle to start from D0 again
    if all(completed_cycle):
        reset_motor_cycles()

    time.sleep(0.1)
    time_step += 1

'''
====================================
8. PLOTTING (disclaimer: used ChatGPT for plotting help!!!)
====================================
'''
# Define colors for each motor, used universally for all plots
colors = {
    'D0': 'red',
    'D1': 'orange',
    'D2': 'yellow',
    'D3': 'green',
    'D4': 'blue',
    'D5': 'purple'
}

# Creates a figure with 6 graphs for each motor's normalized predictions
def create_prediction_plot(data, start_time, end_time, filename):
    fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
    fig.subplots_adjust(hspace=0.3)
    
    # Get time steps and data for the range
    time_slice = slice(start_time, end_time)
    time_steps = data['Time_Step'][time_slice].to_numpy()
    
    # Plot each motor's predictions
    for i, motor_id in enumerate(MOTORS.keys()):
        motor_name = MOTORS[i]['name']
        ax = axes[i]
        
        # Plot prediction line
        predictions = data[f'Pred_{motor_name}'][time_slice].to_numpy()
        ax.plot(time_steps, predictions, color=colors[motor_name], 
                linewidth=2, label='Prediction')
        
        # Highlight regions where motor is active (when position > 0.1 so it is starting to close)
        positions = data[f'Pos_{motor_name}'][time_slice].to_numpy()
        active_regions = positions > 0.1
        
        # Find continuous regions where motor is active
        changes = np.diff(active_regions.astype(int))
        start_indices = np.where(changes == 1)[0] + 1
        end_indices = np.where(changes == -1)[0] + 1
        
        # Add start region if motor starts active
        if active_regions[0]:
            start_indices = np.insert(start_indices, 0, 0)
        # Add end region if motor ends active
        if active_regions[-1]:
            end_indices = np.append(end_indices, len(time_steps) - 1)
        
        # Highlight active regions
        for start, end in zip(start_indices, end_indices):
            if start < len(time_steps) and end < len(time_steps):
                ax.axvspan(time_steps[start], time_steps[end], 
                          color=colors[motor_name], alpha=0.1,
                          label='Finger Activity' if start == start_indices[0] else "")
        
        # Customize plot appearance
        ax.set_ylim(0, 0.5)
        ax.set_ylabel(f'{motor_name}')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add legend to first plot only
        if i == 0:
            ax.legend(loc='upper right')
        
        # Only show x-label for bottom plot
        if i == 5:
            ax.set_xlabel('Time Steps')
    
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

# Creates plot to compare predictions to true return for each motor
def create_comparison_plot(data, start_time, end_time, filename):
    fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
    fig.subplots_adjust(hspace=0.3)
    
    # Get time steps and data for the range
    time_slice = slice(start_time, end_time)
    time_steps = data['Time_Step'][time_slice].to_numpy()
    
    # Create lighter colors for true return lines
    light_colors = {
        'D0': '#ffb3b3',  # light red
        'D1': '#ffd9b3',  # light orange
        'D2': '#fff2b3',  # light yellow
        'D3': '#b3ffb3',  # light green
        'D4': '#b3d9ff',  # light blue
        'D5': '#e6b3ff'   # light purple
    }
    
    # Plot each motor's predictions and true returns
    for i, motor_id in enumerate(MOTORS.keys()):
        motor_name = MOTORS[i]['name']
        ax = axes[i]
        
        predictions = data[f'Pred_{motor_name}'][time_slice].to_numpy()
        true_returns = data[f'TRet_{motor_name}'][time_slice].to_numpy()
        
        # Solid line for predictions
        ax.plot(time_steps, predictions, color=colors[motor_name], 
                linewidth=2, label='Prediction')
        
        # Dashed line for true returns
        ax.plot(time_steps, true_returns, color=light_colors[motor_name], 
                linewidth=2, linestyle='--', label='True Return')
        
        # Customize plot appearance
        ax.set_ylim(0, 0.5)
        ax.set_ylabel(f'{motor_name}')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add legend to first plot only
        if i == 0:
            ax.legend(loc='upper right')
        
        # Only show x-label for bottom plot
        if i == 5:
            ax.set_xlabel('Time Steps')
    
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

# Creates scatter plots of predictions vs normalized posiitons for each motor
def create_state_value_plot(data, start_time, end_time, filename):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Prediction vs Position State Values')
    
    # Get data for the range
    time_slice = slice(start_time, end_time)
    
    # Plot each motor's state-value relationship
    for i, motor_id in enumerate(MOTORS.keys()):
        motor_name = MOTORS[i]['name']
        ax = axes[i//3, i%3]
        
        positions = data[f'NormState_{motor_name}_Pos'][time_slice].to_numpy()
        predictions = data[f'Pred_{motor_name}'][time_slice].to_numpy()
        
        # Create scatter plot
        ax.scatter(positions, predictions, c=colors[motor_name], alpha=0.3, s=10)
        
        ax.set_xlabel('Normalized Position')
        ax.set_ylabel('Prediction')
        ax.set_title(f'{motor_name}')
        # ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim(0, 1)  # Set x-axis range from 0 to 1
        
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

# Creates plot showing prediction error over time for each motor
def create_learning_curve_plot(data, window_size, filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    time_steps = data['Time_Step'].to_numpy()  # Convert to numpy array first
    
    for i, motor_id in enumerate(MOTORS.keys()):
        motor_name = MOTORS[i]['name']
        
        # Calculate running average of absolute prediction error
        pred_error = np.abs(data[f'Pred_{motor_name}'] - data[f'TRet_{motor_name}'])
        error_mean = pd.Series(pred_error).rolling(window=window_size).mean().to_numpy()  # Convert to numpy
        
        # Remove NaN values that come from the rolling window
        valid_indices = ~np.isnan(error_mean)
        ax.plot(time_steps[valid_indices], error_mean[valid_indices], 
                color=colors[motor_name], label=motor_name)
    
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Average Prediction Error')
    # ax.set_title('Learning Curves')
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend()
    
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

# Convert data to pandas DataFrame for easier handling
data = pd.read_csv(csv_filename)

# Generate plots for different time ranges
create_prediction_plot(data, 0, 100, f"{script_filename}_0-100.png")
create_prediction_plot(data, 400, 500, f"{script_filename}_400-500.png")
create_prediction_plot(data, 900, 1000, f"{script_filename}_900-1000.png")
create_prediction_plot(data, 1900, 2000, f"{script_filename}_1900-2000.png")
create_comparison_plot(data, 0, 100, f"{script_filename}_true_return_0-100.png")
create_comparison_plot(data, 400, 500, f"{script_filename}_true_return_400-500.png")
create_comparison_plot(data, 900, 1000, f"{script_filename}_true_return_900-1000.png")
create_comparison_plot(data, 1900, 2000, f"{script_filename}_true_return_1900-2000.png")
create_state_value_plot(data, 0, 2000, f"{script_filename}_state_values.png")
create_learning_curve_plot(data, 50, f"{script_filename}_learning_curves.png")

'''
====================================
9. CLOSING FILES
====================================
'''
# Close the CSV file after writing to it
csv_file.close()

# Disable torque for all motors
for motor_id in MOTORS.keys():
    packetHandler.write1ByteTxRx(portHandler, motor_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
portHandler.closePort()