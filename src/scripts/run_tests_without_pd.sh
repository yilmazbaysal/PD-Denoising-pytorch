python ../test_cpu_without_pd.py\
 --scale 1\
 --ps 2 --ps_scale 2\
 --real_n 2\
 --k 0.5\
 --mode CBD\
 --color 1\
 --output_map 0\
 --zeroout 0 --keep_ind 0\
 --num_of_layers 20\
 --delog ../../logs\
 --cond 1 --refine 0 --refine_opt 1\
 --test_data qualitative/noisy\
 --test_data_gnd qualitative/ground_truth\
 --out_dir ../../results/test