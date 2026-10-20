"""Run: python -m X02.write --text 永 --hold."""
from PIPER.write import main as run_writing
from .env import X02WritingEnv, HERE, MODEL, REFERENCE
from .trajectory import plan


def main():
    run_writing(env_factory=X02WritingEnv, planner=plan, here=HERE,
                reference=REFERENCE, model_file=MODEL, center=(.20, -.25),
                height=-.23, home_deg=(0.,)*7, metadata_name='x02_writing_data.json',
                camera_distance=1.9, camera_height=.2)


if __name__ == '__main__':
    main()
