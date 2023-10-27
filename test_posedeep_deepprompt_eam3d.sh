CUDA_VISIBLE_DEVICES=0 python test_posedeep_deepprompt_eam3d.py --name deepprompt_eam3d_st_tanh_304_3090_all\ 24_10_23_13.11.40 --part 0 --mode 1 &
CUDA_VISIBLE_DEVICES=1 python test_posedeep_deepprompt_eam3d.py --name deepprompt_eam3d_st_tanh_304_3090_all\ 24_10_23_13.11.40 --part 2 --mode 1 &
CUDA_VISIBLE_DEVICES=0 python test_posedeep_deepprompt_eam3d.py --name deepprompt_eam3d_st_tanh_304_3090_all\ 24_10_23_13.11.40 --part 1 --mode 1 &
CUDA_VISIBLE_DEVICES=1 python test_posedeep_deepprompt_eam3d.py --name deepprompt_eam3d_st_tanh_304_3090_all\ 24_10_23_13.11.40 --part 3 --mode 1 &
wait
