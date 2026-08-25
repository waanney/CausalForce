# Multiple Coexisting Risks Dataset

## Naming Scheme

### Occlusion (Occ_xx)
|  ID    | Description                               |
| ------ | ------------------------------------------|
| Occ_01 | EgoForwardOccluderPullOver                |
| Occ_02 | EgoForwardOccluderSuddenBrake             |
| Occ_03 | EgoLeftTurnOccluderWaitingInJunction      |
| Occ_04 | EgoRightTurnLaneChangeOccluderYielding    |
| Occ_05 | EgoRightTurnOccluderYielding              |

### Interaction (I_xx)
| ID   | Description                                     |
| ---- | ------------------------------------------------|
| I_01 | EgoRightTurnActorForward                        |
| I_02 | EgoForwardActorForward                          |
| I_03 | EgoLeftTurnActorForward                         |
| I_04 | EgoLaneChangeRightActorForward                  |
| I_05 | EgoLaneChangeLeftActorForward                   |
| I_06 | EgoRightTurnActorRightTurn                      |
| I_07 | EgoForwardActorRightTurn                        |
| I_08 | EgoLeftTurnActorRightTurn                       |
| I_09 | EgoRightTurnActorLeftTurn                       |
| I_10 | EgoForwardActorLeftTurn                         |
| I_11 | EgoLeftTurnActorLeftTurn                        |
| I_12 | EgoForwardActorLaneChangeRight                  |
| I_13 | EgoForwardActorLaneChangeLeft                   |

### Collision (C_xx)
| ID   | Description                                     |
| ---- | ------------------------------------------------|
| C_01 | EgoRightTurnActorForward                        |
| C_02 | EgoForwardActorForward                          |
| C_03 | EgoLeftTurnActorForward                         |
| C_04 | EgoRightTurnActorRightTurn                      |
| C_05 | EgoForwardActorRightTurn                        |
| C_06 | EgoLeftTurnActorRightTurn                       |
| C_07 | EgoRightTurnActorLeftTurn                       |
| C_08 | EgoForwardActorLeftTurn                         |
| C_09 | EgoLeftTurnActorLeftTurn                        |
| C_10 | EgoForwardActorLaneChangeRight                  |
| C_11 | EgoForwardActorLaneChangeLeft                   |

### Obstacle (Obs_xx)
| ID     | Description                |
| ------ | ---------------------------|
| Obs_01 | EgoLaneChangeLeft          |
| Obs_02 | EgoLaneChangeRight         |
| Obs_03 | EgoRightTurnLaneChangeLeft |
| Obs_04 | EgoLeftTurnLaneChangeRight |



## Dataset Structure
```
rgb_front: front-view camera images at 900x256 resolution
seg_front: corresponding segmentation images
depth_front: corresponding depth images
birdview: topdown segmentation images
2d_bbs_front: 2d bounding boxes for different agents in the corresponding camera view
3d_bbs: 3d bounding boxes for different agents
affordances: different types of affordances
measurements: contains ego-agent position, velocity and other metadata
actors_data: contains the positions, velocities and other metadatas of surrounding vehicles and the traffic lights
risk_interval_H8_new: risk interval annotations for risk objects
trajectory_H8_normalized: objects' 2D trajectory in front-view images
risk_id.json: risk objects' category and ID
risk_interval.json: risk objects’ begin and end frames of risk 
```
