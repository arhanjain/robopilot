import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--env', required=True, type=str)
args = argparser.parse_args()

import zerorpc

if __name__ == "__main__":
    env = None
    match args.env:
        case "mujoco":
            from envs.mujoco import MujocoSim
            env = MujocoSim()
        case "real":
            from envs.real import RealWorld
            env = RealWorld()
        case _:
            raise ValueError("Unknown env: {}".format(args.env))
            
    server = zerorpc.Server(env)
    server.bind("tcp://0.0.0.0:4242")
    server.run()


