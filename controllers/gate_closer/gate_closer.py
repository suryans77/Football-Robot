from controller import Robot

robot = Robot()
# Basic simulation time step in milliseconds
timestep = int(robot.getBasicTimeStep())

# 1. INITIALIZE DEVICES
# Fixed: Added quotes around device names
left_door = robot.getDevice("left_door")
right_door = robot.getDevice("right_door")

# Chassis motors
# Fixed: Added quotes around device names
left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# --- TEST PARAMETERS ---
# Match this to the minStop/maxStop in your PROTO
SLIDE_DISTANCE = 0.025

# Fixed: Added quotes for the print statements
print("--- MECHANICAL GATE TEST STARTING ---")
print("Doors should cycle every 2 seconds.")

while robot.step(timestep) != -1: # Fixed: Added missing colon
    # Use simulation time to create a 4-second cycle
    current_time = robot.getTime()
    
    # First 2 seconds: CLOSE DOORS
    # Fixed: Added missing comparison operator (<) and colon
    if (current_time % 4.0) < 2.0:
        # Move doors to the center
        left_door.setPosition(-SLIDE_DISTANCE)
        right_door.setPosition(SLIDE_DISTANCE)
        # Fixed: Added missing multiplication (*) and colon
        if int(current_time * 10) % 40 == 0:
            print(f"Time: {current_time:.1f}s -> ACTION: CLOSING")
            
    # Next 2 seconds: OPEN DOORS
    else:
        # Return to home position
        left_door.setPosition(0.0)
        right_door.setPosition(0.0)
        # Fixed: Added missing multiplication (*)
        if int(current_time * 10) % 40 == 0:
            print(f"Time: {current_time:.1f}s -> ACTION: OPENING")