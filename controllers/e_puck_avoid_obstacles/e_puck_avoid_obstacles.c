/*
 * EPuckPlus Avoid Obstacles Controller
 * Adapted from the original e-puck controller
 * Compatible with EPuckPlus physics (higher mass, speed, inertia)
 */

#include <stdio.h>
#include <stdlib.h>

#include <webots/distance_sensor.h>
#include <webots/led.h>
#include <webots/motor.h>
#include <webots/robot.h>

/* --- Constants --- */
#define DISTANCE_SENSORS_NUMBER 8
#define LEDS_NUMBER 10

#define LEFT 0
#define RIGHT 1

/* EPuckPlus tuned values */
#define MAX_SPEED 10.0        // rad/s (motor command cap)
#define MAX_ACCEL 3.0         // rad/s^2 (smooth motion)

/* --- Devices --- */
static WbDeviceTag distance_sensors[DISTANCE_SENSORS_NUMBER];
static double distance_sensors_values[DISTANCE_SENSORS_NUMBER];
static const char *distance_sensors_names[DISTANCE_SENSORS_NUMBER] =
  {"ps0", "ps1", "ps2", "ps3", "ps4", "ps5", "ps6", "ps7"};

static WbDeviceTag leds[LEDS_NUMBER];
static bool leds_values[LEDS_NUMBER];
static const char *leds_names[LEDS_NUMBER] =
  {"led0", "led1", "led2", "led3", "led4", "led5", "led6", "led7", "led8", "led9"};

static WbDeviceTag left_motor, right_motor;

/* --- Control variables --- */
static double speeds[2] = {0.0, 0.0};
static double prev_speeds[2] = {0.0, 0.0};

/* --- Braitenberg parameters (unchanged logic) --- */
static double weights[DISTANCE_SENSORS_NUMBER][2] = {
  {-1.3, -1.0}, {-1.3, -1.0}, {-0.5,  0.5}, {0.0,  0.0},
  { 0.0,  0.0}, { 0.05,-0.5}, {-0.75, 0.0}, {-0.75, 0.0}
};

static double offsets[2] = {0.25 * MAX_SPEED, 0.25 * MAX_SPEED};

/* --- Utility --- */
static int get_time_step() {
  static int time_step = -1;
  if (time_step < 0)
    time_step = (int)wb_robot_get_basic_time_step();
  return time_step;
}

static double limit_accel(double target, double current, double dt) {
  double max_delta = MAX_ACCEL * dt;
  if (target > current + max_delta)
    return current + max_delta;
  if (target < current - max_delta)
    return current - max_delta;
  return target;
}

/* --- Initialization --- */
static void init_devices() {
  int i;

  for (i = 0; i < DISTANCE_SENSORS_NUMBER; i++) {
    distance_sensors[i] = wb_robot_get_device(distance_sensors_names[i]);
    wb_distance_sensor_enable(distance_sensors[i], get_time_step());
  }

  for (i = 0; i < LEDS_NUMBER; i++)
    leds[i] = wb_robot_get_device(leds_names[i]);

  left_motor = wb_robot_get_device("left wheel motor");
  right_motor = wb_robot_get_device("right wheel motor");

  wb_motor_set_position(left_motor, INFINITY);
  wb_motor_set_position(right_motor, INFINITY);
  wb_motor_set_velocity(left_motor, 0.0);
  wb_motor_set_velocity(right_motor, 0.0);
}

/* --- Sensors --- */
static void read_sensors() {
  int i;
  for (i = 0; i < DISTANCE_SENSORS_NUMBER; i++) {
    distance_sensors_values[i] =
      wb_distance_sensor_get_value(distance_sensors[i]) / 4096.0;
  }
}

/* --- Behavior --- */
static void run_braitenberg() {
  int i, j;

  for (i = 0; i < 2; i++) {
    speeds[i] = offsets[i];
    for (j = 0; j < DISTANCE_SENSORS_NUMBER; j++)
      speeds[i] += distance_sensors_values[j] * weights[j][i] * MAX_SPEED;

    if (speeds[i] >  MAX_SPEED) speeds[i] =  MAX_SPEED;
    if (speeds[i] < -MAX_SPEED) speeds[i] = -MAX_SPEED;
  }
}

/* --- Actuators --- */
static void update_actuators() {
  int i;
  double dt = get_time_step() / 1000.0;

  for (i = 0; i < LEDS_NUMBER; i++)
    wb_led_set(leds[i], leds_values[i]);

  speeds[LEFT]  = limit_accel(speeds[LEFT],  prev_speeds[LEFT],  dt);
  speeds[RIGHT] = limit_accel(speeds[RIGHT], prev_speeds[RIGHT], dt);

  wb_motor_set_velocity(left_motor,  speeds[LEFT]);
  wb_motor_set_velocity(right_motor, speeds[RIGHT]);

  prev_speeds[LEFT]  = speeds[LEFT];
  prev_speeds[RIGHT] = speeds[RIGHT];
}

static void blink_leds() {
  static int counter = 0;
  counter++;
  leds_values[(counter / 10) % LEDS_NUMBER] = true;
}

/* --- Main --- */
int main() {
  wb_robot_init();

  printf("EPuckPlus avoid obstacles controller started\n");

  init_devices();

  while (wb_robot_step(get_time_step()) != -1) {
    int i;
    for (i = 0; i < LEDS_NUMBER; i++)
      leds_values[i] = false;

    read_sensors();
    blink_leds();
    run_braitenberg();
    update_actuators();
  }

  wb_robot_cleanup();
  return 0;
}
