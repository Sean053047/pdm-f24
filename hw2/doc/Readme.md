Required package:  
opencv-python
numpy  
pandas  
scikit-learn
scipy

# Usage  

Make sure that 
1. ```apartment_0``` folder  
2. ```color_coding_semantic_segmentation_classes.xlsx```  
3. ```semantic_3d_pointcloud``` folder

All of them are located at ```HW2``` folder. Instead of that, you also can specify option, ```--test_scene```, ```--test_scene_json```, to get the corresponding ```mesh_semantic.ply``` and ```info_semantic.json```.

RRT Mode:  
rrt_mode = "rrt";  sample_mode = "random" or "goal_bias"

RRT* MOde:
rrt_mode = "rrt_star"; sample_mode = "random" or "goal_bias" or "diverse"  

```bash
cd ${HW2 DIRECTORY}

# For General usage
python3 src/main.py --step ${STEP SIZE} --max_dist ${MAX DISTANCE} -cm --rrt_mode ${RRT MODE} --sample_mode ${SAMPLE MODE} --output_folder ${OUTPUT FOLDER} --test_scene ${ROUTE for MESH_SEMANTIC} --test_scene_json ${SEMANTIC INFO JSON} # Optional: --refine

# For RRT
python3 src/main.py --step 50 --max_dist 120 -cm --rrt_mode rrt --sample_mode random --output_folder ${OUTPUT FOLDER} --test_scene ${ROUTE for MESH_SEMANTIC} --test_scene_json ${SEMANTIC INFO JSON}

# For RRT*
python3 src/main.py --step 50 --max_dist 120 -cm --rrt_mode rrt_star --sample_mode diverse --refine --output_folder ${OUTPUT FOLDER} --test_scene ${ROUTE for MESH_SEMANTIC} --test_scene_json ${SEMANTIC INFO JSON}
```
