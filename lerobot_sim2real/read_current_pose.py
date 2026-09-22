from real_robot import create_real_robot


def main():
    robot = create_real_robot("so101")
    robot.connect(calibrate=False)
    try:
        obs = robot.get_observation()
        print("Current joint positions (degrees, calibrated):")
        for key, val in obs.items():
            if key.endswith(".pos"):
                print(f"  {key:20s} {val:8.2f}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
