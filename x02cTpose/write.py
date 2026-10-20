"""Four-axis T-pose right-arm Chinese/English writing simulation."""
from PIPER.write import main as run_writing
from .env import TPoseWritingEnv, HERE, MODEL, REFERENCE
from .trajectory import plan


def main():
    print('Four active joints, fixed wrist: position tracking with soft tilt minimization; vertical pen pose is not guaranteed.',flush=True)
    run_writing(env_factory=TPoseWritingEnv,planner=plan,here=HERE,
                reference=REFERENCE,model_file=MODEL,center=(.30,-.20),height=.90,
                home_deg=(0.,)*4,metadata_name='tpose_writing_data.json',
                camera_distance=2.6,camera_height=.95)


if __name__=='__main__':main()
