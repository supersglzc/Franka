import os

from gymnasium.envs.registration import register

with open(os.path.join(os.path.dirname(__file__), "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()

ENV_IDS = []

for task in ["Reach", "Slide", "Push", "PickAndPlace", "Stack", "Flip",
             "PegInsertion", "Drawer", "DrawerMulti", "Cabinet"]:
    for reward_type in ["sparse", "dense"]:
        for control_type in ["ee", "joints"]:

            reward_suffix = "Dense" if reward_type == "dense" else ""
            control_suffix = "Joints" if control_type == "joints" else ""
            env_id = f"Panda{task}{control_suffix}{reward_suffix}-v3"

            register(
                id=env_id,
                entry_point=f"panda_gym.envs:Panda{task}Env",
                kwargs={"reward_type": reward_type, "control_type": control_type},
                max_episode_steps=100,
            )

            # adding randomize starting point to Reach and PegInsertion task
            if task in ["Reach", "PegInsertion"]:
                random_suffix = 'Random'
                reward_suffix = "Dense" if reward_type == "dense" else ""
                control_suffix = "Joints" if control_type == "joints" else ""
                env_id = f"Panda{task}{control_suffix}{reward_suffix}{random_suffix}-v3"

                register(
                    id=env_id,
                    entry_point=f"panda_gym.envs:Panda{task}Env",
                    kwargs={"reward_type": reward_type, "control_type": control_type, "random_init_pos": True},
                    max_episode_steps=100,
                )

            ENV_IDS.append(env_id)
