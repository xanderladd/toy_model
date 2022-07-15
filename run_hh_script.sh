export OMP_NUM_THREADS=1
srun -n 640 python ./run_volts_hh/run_stim_hdf5.py 
srun -n 640 python ./score_volts_efficent_sandbox/score_volts_hdf5_efficent_sandbox.py 
srun -n 640 python ./analyze_p/analyze_p_parallel.py
