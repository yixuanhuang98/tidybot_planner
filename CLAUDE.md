I want to build a general data collection and training pipeline for the current environment, such as running with 
```
python3 main.py  --sim --motion_planner_table_stack --save-images
```
currently it shows the demo. 

I want to implement with lerobot (@lerobot/lerobot), 
* data collection pipeline that collects in lerobot format 
* training pipeline that trains a model, such as diffusion policy, with the collected data