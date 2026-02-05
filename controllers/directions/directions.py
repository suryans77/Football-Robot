from controller import Robot
import math

robot = Robot()
timestep = int(robot.getBasicTimeStep())
compass = robot.getDevice("compass")
compass.enable(timestep)

left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

# Spin slowly to capture data
left_motor.setVelocity(2.0)
right_motor.setVelocity(-2.0)

print("--- STARTING COMPASS CALIBRATION SPIN ---")

while robot.step(timestep) != -1:
    cv = compass.getValues()
    
    # Calculate angles using different atan2 combinations
    # One of these will likely align with your intuition
    angle_yx = math.degrees(math.atan2(cv[1], cv[0]))
    angle_xy = math.degrees(math.atan2(cv[0], cv[1]))
    
    print(f"X: {cv[0]:.2f} | Y: {cv[1]:.2f} | Angle(Y,X): {angle_yx:.1f}° | Angle(X,Y): {angle_xy:.1f}°")