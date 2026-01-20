from controller import Robot

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Devices
camera = robot.getDevice("camera")
camera.enable(timestep)

left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

MAX_SPEED = 15.0 # Using your agile parameters

while robot.step(timestep) != -1:
    image = camera.getImage()
    width = camera.getWidth()
    height = camera.getHeight()

    ball_x = -1
    pixels_found = 0
    sum_x = 0

    # Scan the image for Yellow/Orange pixels
    for x in range(width):
        for y in range(height):
            # Webots getImage returns a BGRA byte string
            # Indexing: 4 * (y * width + x)
            r = camera.imageGetRed(image, width, x, y)
            g = camera.imageGetGreen(image, width, x, y)
            b = camera.imageGetBlue(image, width, x, y)

            # Color Logic: High Red + High Green + Low Blue = Yellow/Orange
            if r > 150 and g > 100 and b < 80:
                sum_x += x
                pixels_found += 1

    if pixels_found > 0:
        # Calculate the horizontal center of the ball
        ball_x = sum_x / pixels_found
        
        # Determine deviation from center (0.0 is center, -1.0 is left, 1.0 is right)
        deviation = (ball_x / width) - 0.5
        
        # Human-like steering logic
        if abs(deviation) < 0.1:
            # Ball is centered -> Sprint forward
            left_speed = MAX_SPEED
            right_speed = MAX_SPEED
        elif deviation < 0:
            # Ball is to the left -> Turn left
            left_speed = MAX_SPEED * 0.2
            right_speed = MAX_SPEED
        else:
            # Ball is to the right -> Turn right
            left_speed = MAX_SPEED
            right_speed = MAX_SPEED * 0.2
    else:
        # Ball lost -> Pivot on spot to find it
        left_speed = MAX_SPEED * 0.5
        right_speed = -MAX_SPEED * 0.5

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)