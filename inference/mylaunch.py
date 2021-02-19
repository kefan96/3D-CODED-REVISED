import os
import gpustat
import time

def get_first_available_gpu():
    """
    Check if a gpu is free and returns it
    :return: gpu_id
    """
    query = gpustat.new_query()
    for gpu_id in range(len(query)):
        gpu = query[gpu_id]
        if gpu.memory_used < 700:
            has = os.system("tmux has-session -t " + f"GPU{gpu_id}" + " 2>/dev/null")
            if not int(has)==0:
                return gpu_id
    return -1

def job_scheduler(dict_of_jobs):
    """
    Launch Tmux session each time it finds a free gpu
    :param dict_of_jobs:
    """
    keys = list(dict_of_jobs.keys())
    while len(keys) > 0:
        job_key = keys.pop()
        job = dict_of_jobs[job_key]
        while get_first_available_gpu() < 0:
            print("Waiting to find a GPU for ", job)
            time.sleep(30) # Sleeps for 30 sec
        gpu_id = get_first_available_gpu()
        cmd = f"conda activate /scratch/ky1323/3D-CODED/envs; CUDA_VISIBLE_DEVICES={gpu_id} {job} 2>&1 | tee  log_terminals/{gpu_id}_{job_key}.txt; tmux kill-session -t GPU{gpu_id}"
        CMD = f'tmux new-session -d -s GPU{gpu_id} \; send-keys "{cmd}" Enter'
        print(CMD)
        os.system(CMD)
        time.sleep(15)  # Sleeps for 30 sec