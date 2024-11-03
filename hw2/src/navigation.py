import numpy as np
from PIL import Image
import numpy as np
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import cv2
from scipy.spatial.transform import Rotation

# This is the scene we are going to load.
# support a variety of mesh formats, such as .glb, .gltf, .obj, .ply
### put your scene path ###
test_scene = "replica_v1/apartment_0/habitat/mesh_semantic.ply"

sim_settings = {
    "scene": test_scene,  # Scene path
    "default_agent": 0,  # Index of the default agent
    "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
    "width": 512,  # Spatial resolution of the observations
    "height": 512,
    "sensor_pitch": 0,  # sensor pitch (x rotation in rads)
}

# This function generates a config for the simulator.
# It contains two parts:
# one for the simulator backend
# one for the agent, where you can attach a bunch of sensors

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def transform_semantic(semantic_obs, id2label):
    id2label = np.array(id2label, dtype=np.uint32)
    SEMANTIC_LABEL = id2label[semantic_obs]
    return SEMANTIC_LABEL

def make_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    #semantic snesor
    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    agent_cfg.sensor_specifications = [rgb_sensor_spec, semantic_sensor_spec]
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=settings['action_move']),
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=settings['action_turn'])  # In degree
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=settings['action_turn'])
        ),
    }
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def show_observation(observations, tt=0):
    cv2.imshow("RGB", transform_rgb_bgr(observations["color_sensor"]))
    cv2.waitKey(tt)


def get_agent_state(loc, ex_rot):
    '''
    loc = [x, y, z]
    rotation = [ rx, ry, rz] in Euler angles, radians.
    '''
    R = Rotation.from_euler('xyz', ex_rot, degrees=False)
    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array(loc)  # agent in world space
    agent_state.rotation = R.as_quat()
    return agent_state

def get_ry_from_vec(vec1, vec2, degrees=False):
    '''
    vector = [z, x]
    '''
    theta = np.arccos(np.dot(vec1,vec2) / (np.linalg.norm(vec1)* np.linalg.norm(vec2)))
    theta = theta if np.cross(vec1,vec2) > 0 else -theta
    if np.all(np.isnan(theta)):
        print(f"Rot NAN: {vec1}, {vec2}")
        theta = 0
    return theta*180/np.pi if degrees else theta

def navigateAndSee(sim, agent, action="", target=None, SEMANTIC_INFO=None, tt=2):
    assert (target is not None and SEMANTIC_INFO is not None) or  target is None
    observations = sim.step(action)
    semantic_img = transform_semantic(observations['semantic_sensor'], SEMANTIC_INFO['id_to_label'])
    tgt_label = [v["id"] for v in SEMANTIC_INFO['classes'] if v['name'] == target][0]
    tgt_mask  = semantic_img == tgt_label
    color_img = transform_rgb_bgr(observations["color_sensor"])
    
    if np.sum(tgt_mask) >0:
        color_img = np.where(
            tgt_mask[:, :, np.newaxis], 
            (color_img*0.5 + np.array([0, 0, 255]).reshape((1,1,3)) * 0.5).astype(np.uint8),
            color_img
        )
    cv2.imshow("color_sensor", color_img )
    if tt is not None:
        cv2.waitKey(tt)
    return color_img

def manipulate_agent(sim , agent , target, SEMANTIC_INFO):
    navigateAndSee(sim, agent, "turn_left", target, SEMANTIC_INFO, tt=1)
    FORWARD_KEY="w"
    LEFT_KEY="a"
    RIGHT_KEY="d"
    FINISH="f"

    while True:
        keystroke = cv2.waitKey(0)
        if keystroke == ord(FORWARD_KEY):
            action = "move_forward"
            navigateAndSee(sim, agent, action, target, SEMANTIC_INFO, tt=1)
            print("action: FORWARD")
        elif keystroke == ord(LEFT_KEY):
            action = "turn_left"
            navigateAndSee(sim, agent, action, target, SEMANTIC_INFO, tt=1)
            print("action: LEFT")
        elif keystroke == ord(RIGHT_KEY):
            action = "turn_right"
            navigateAndSee(sim, agent, action, target, SEMANTIC_INFO, tt=1)
            print("action: RIGHT")
        elif keystroke == ord(FINISH):
            print("action: FINISH")
            break
        else:
            print("INVALID KEY")
            continue
    