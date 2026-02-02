from controller import Robot

# Initialize the Robot
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Get and initialize motors
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')

# Set motors to infinity mode to control by velocity
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

# Set the spin speed (rad/s)
# Opposite values cause the robot to pivot on its center
spin_speed = 2.0

left_motor.setVelocity(spin_speed)
right_motor.setVelocity(-spin_speed)

# Main simulation loop
while robot.step(timestep) != -1:
    pass