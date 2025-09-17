import traci
import sumolib
import numpy as np
import torch
import utils
import TD3
import custom_td3
import custom_replay_buffer

NUM_LANES = 3
CONTROL_DISTANCE = 100
MAX_SPEED = 13.9
VSTATE_DIM = 14
ACTION_DIM = 1
HIDDEN_LAYERS = 256


def one_hot_encode(i, n):
    vec = np.zeros(n, dtype=int)
    vec[i] = 1
    return vec

def one_hot_encode_way(fid):
    if fid in ("WE", "SN", "EW", "NS"):
        return np.array([1, 0, 0])
    if fid in ("WS", "SE", "EN", "NW"):
        return np.array([0, 1, 0])
    
    return np.array([0, 0, 1])

def one_hot_encode_queue(d):
    if d == "S": return np.array([1, 0, 0, 0])
    if d == "N": return np.array([0, 1, 0, 0])
    if d == "W": return np.array([0, 0, 1, 0])
    return np.array([0, 0, 0, 1])


def normalize(x, m, M):
    return 2 * (x - m) / (M - m) - 1

def denormalize(x_norm, m, M):
    return (x_norm + 1) / 2 * (M - m) + m

def get_state():
    state = []
    vehicule_done = {}
    vids = []

    vehicle_ids = traci.vehicle.getIDList()
    collided_ids = traci.simulation.getCollidingVehiclesIDList()

    for vid in vehicle_ids:
        pos = traci.vehicle.getPosition(vid)

        relative_pos_x = pos[0] - junction_center[0]
        relative_pos_y = pos[1] - junction_center[1]

        # Do not conisder vehicles further than 100m from the intersection
        if np.sqrt(relative_pos_x**2 + relative_pos_y**2) > CONTROL_DISTANCE:
            continue

        vids.append(vid)

        speed = traci.vehicle.getSpeed(vid)               
        angle = traci.vehicle.getAngle(vid)

        vehicule_done[vid] = True if traci.vehicle.getRoadID(vid)[0] == 'O' or vid in collided_ids else False
        
        lid = traci.vehicle.getLaneIndex(vid)
        fid = vid.split(".")[0]

        # Prepare vehicle state vector
        relative_pos_x = normalize(relative_pos_x, -CONTROL_DISTANCE, CONTROL_DISTANCE)
        relative_pos_y = normalize(relative_pos_y, -CONTROL_DISTANCE, CONTROL_DISTANCE)
        speed = normalize(speed, 0, MAX_SPEED)
        angle = normalize(angle, 0, 360)
        lane = one_hot_encode(lid, NUM_LANES)
        way = one_hot_encode_way(fid)
        queue = one_hot_encode_queue(fid[0])

        vstate = np.concatenate([
            [relative_pos_x, relative_pos_y, speed, angle], 
            lane, 
            way, 
            queue
        ])

        state.append(vstate)
    
    state = np.array(state)

    return state, vehicule_done, collided_ids, vids









# Obtain coordinates of the center of the intersection
network_path = "intersection.net.xml"
network = sumolib.net.readNet(network_path)
junction = network.getNode("O")
junction_center = junction.getCoord()


# TD3 parameters as per the paper. Non specified will be take the default TD3 parameters
seed = 0 # should be made random eventually
start_timesteps = 25e-3
eval_freq = 5e3
max_timesteps = 1e6
expl_noise=0.1

# Parameters specified by the paper
batch_size=128
discount=0.99
tau=4e-3
policy_noise=0.2
noise_clip=0.3
policy_freq=2

torch.manual_seed(seed)
np.random.seed(seed)

state_dim = VSTATE_DIM # the encoded state is the real state seen by the Actor-Critic
action_dim = ACTION_DIM
max_action=1

kwargs = {
    "n_features":VSTATE_DIM,
    "hidden_size": HIDDEN_LAYERS,
    "action_dim":action_dim,
    "max_action": max_action,
    "discount":discount,
    "tau":tau,
    "policy_noise":policy_noise*max_action,
    "noise_clip": noise_clip*max_action,
    "policy_freq": policy_freq
}

def eval_policy(select_action, steps=1000, random=False):
    traci.start(["sumo-gui", "-c", sumo_config, "--collision.check-junctions"])

    step = 0
    active_rewards = {}   
    n_finished = 0
    avg_reward = 0.0   
    

    while step < steps:
        traci.simulationStep()

        vehicle_ids = traci.vehicle.getIDList()
        state, done, collided = get_state()

        vehicle_inputs = []
        vehicle_ids_to_update = []

        for i, vid in enumerate(vehicle_ids):
            if done.get(vid, False):
                continue

            if vid not in active_rewards:
                active_rewards[vid] = 0
        
            ego_state = state[i]
            other_states = [s for j, s in enumerate(state) if j != i]
            vehicle_input = np.vstack([ego_state] + other_states)
            vehicle_inputs.append(vehicle_input)
            vehicle_ids_to_update.append(vid)

            active_rewards[vid] -= 1

        actions = policy.select_action(vehicle_inputs)

        for vid, action in zip(vehicle_ids_to_update, actions):
            traci.vehicle.setSpeedMode(vid, 96)  # fully manual
            traci.vehicle.setSpeed(vid, denormalize(action[0], 0, 13.9))

        finished_vids = [vid for vid in active_rewards if done.get(vid, False)]
        for vid in finished_vids:
            if vid in collided:
                active_rewards[vid] -= 100
            else:
                active_rewards[vid] += 100

            print(active_rewards[vid])

            n_finished += 1
            avg_reward += (active_rewards[vid] - avg_reward) / n_finished

            del active_rewards[vid]

        step += 1

    traci.close()

    return avg_reward


def train_policy(
    policy,
    replay_buffer,
    steps=100_000,
    batch_size=128,
    start_timesteps=1_000
):
    traci.start(["sumo", "-c", sumo_config, "--collision.check-junctions"])

    step = 0
    active_rewards = {}  # track per-vehicle cumulative reward

    # initial environment state
    traci.simulationStep()
    state, done, collided = get_state()

    while step < steps:
        vehicle_ids = traci.vehicle.getIDList()

        vehicle_inputs = []
        vehicle_ids_to_update = []

        # build per-vehicle states
        for i, vid in enumerate(vehicle_ids):
            if done.get(vid, False):
                continue

            if vid not in active_rewards:
                active_rewards[vid] = 0

            print(state)
            ego_state = state[i]
            other_states = [s for j, s in enumerate(state) if j != i]
            vehicle_input = np.vstack([ego_state] + other_states).flatten()

            vehicle_inputs.append(vehicle_input)
            vehicle_ids_to_update.append(vid)

        # pick actions
        if step < start_timesteps:
            # pure random exploration
            actions = np.random.uniform(0.0, 1.0, size=(len(vehicle_inputs), 1))
        else:
            actions = policy.select_action(vehicle_inputs)
            # exploration noise
            actions += np.random.normal(0, 0.1, size=actions.shape)
            actions = np.clip(actions, 0.0, 1.0)

        # apply actions
        for vid, state_input, action in zip(vehicle_ids_to_update, vehicle_inputs, actions):
            traci.vehicle.setSpeedMode(vid, 96)  # manual control
            traci.vehicle.setSpeed(vid, denormalize(action[0], 0, 13.9))

        # step environment
        traci.simulationStep()
        next_state, done, collided = get_state()

        # store transitions
        for i, vid in enumerate(vehicle_ids_to_update):
            # build next state for this vehicle
            ego_next = next_state[i]
            others_next = [s for j, s in enumerate(next_state) if j != i]
            next_state_input = np.vstack([ego_next] + others_next).flatten()

            # reward shaping
            reward = -1
            if done.get(vid, False):
                reward = -100 if vid in collided else 100

            replay_buffer.add(
                state=vehicle_inputs[i],
                action=actions[i],
                next_state=next_state_input,
                reward=reward,
                done=done.get(vid, False)
            )

        # train policy if enough data
        if step >= start_timesteps and replay_buffer.size >= batch_size:
            batch = replay_buffer.sample(batch_size)
            policy.train(batch)

        step += 1

    traci.close()
    policy.save("trained_model")


replay_buffer = utils.ReplayBuffer()

policy = TD3.TD3(**kwargs)

sumo_config = "simulation.sumocfg"

# avg_reward = train_policy(policy, replay_buffer)
# print(avg_reward)











device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
traci.start(["sumo", "-c", sumo_config, "--collision.check-junctions"])
step = 0

policy = custom_td3.TD3(14, 256, 1, 1e-5, 1e-4)
replay_buffer = custom_replay_buffer.ReplayBuffer()

start_timesteps = 100
steps = 1000
step = 0

traci.simulationStep()
state, _, _, vids = get_state()
state = utils.sort_state(torch.FloatTensor(state).to(device))

while step < steps:
    
    actions = {}
    for (i, vid) in enumerate(vids):
        
        state_i = utils.move_ego_to_front(state, i)
        if (step < start_timesteps):
            action = np.random.uniform(-1, 1)
        else:
            action = policy.select_action(state_i.unsqueeze(0))

        traci.vehicle.setSpeedMode(vid, 96)
        traci.vehicle.setSpeed(vid, denormalize(action, 0, 13.9))

        actions[vid] = action
    
    traci.simulationStep()

    next_state, _, _, vids = get_state()
    next_state = utils.sort_state(torch.FloatTensor(next_state).to(device))


    state = next_state

    step += 1




# actor = policy.actor
# critic = policy.critic

# traci.simulationStep()
# state1, _, _ = get_state()

# traci.simulationStep()
# state2, _, _ = get_state()

# state1 = torch.FloatTensor(state1).to(device)
# state2 = torch.FloatTensor(state2).to(device)

# state1 = utils.sort_state(state1)
# state2 = utils.sort_state(state2)

# state1 = utils.move_ego_to_front(state1, 2)
# state2 = utils.move_ego_to_front(state2, 2)


# batch = [state1,  
#          state2]


# val = actor(batch)
# cvals = critic(batch, val)

# print(val, cvals)

traci.close()