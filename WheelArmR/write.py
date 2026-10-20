"""Run from repository root: python -m WheelArmR.write --text 永 --hold."""
from PIPER.write import main as run_writing
from .env import WheelArmWritingEnv, HERE, REFERENCE, MODEL
from .trajectory import plan


def main():
    run_writing(env_factory=WheelArmWritingEnv, planner=plan, here=HERE,
                reference=REFERENCE, model_file=MODEL, center=(.35, -.4),
                home_deg=(0.,)*7, metadata_name="wheelarm_writing_data.json")


if __name__ == "__main__":
    main()
