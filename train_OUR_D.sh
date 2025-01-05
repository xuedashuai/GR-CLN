# !/bin/bash
echo " Running Training EXP"

# CUDA_VISIBLE_DEVICES=4 python3 train_OUR_D.py --lr 0.001 --n_stgcnn 1 --n_txpcnn 5  --dataset Null --tag social-stgcnn-ODMAT --use_lrschd --num_epochs 250 --tasks 1 --mem_size 3500 --margin 0.5 --eps 0.001 --cur_task 0 && echo "MA scenarios training Launched." 

# 2-FT
# CUDA_VISIBLE_DEVICES=5 python3 train_OUR_D.py --lr 0.001 --n_stgcnn 1 --n_txpcnn 5  --dataset Null --tag social-stgcnn-ODFTE --use_lrschd --num_epochs 250 --tasks 2 --mem_size 3500 --margin 0.5 --eps 0.001 --cur_task 1 && echo "MA scenarios training Launched." 

# 3-ZS
# CUDA_VISIBLE_DEVICES=5 python3 train_OUR_D.py --lr 0.001 --n_stgcnn 1 --n_txpcnn 5  --dataset Null --tag social-stgcnn-ODZS --use_lrschd --num_epochs 250 --tasks 3 --mem_size 3500 --margin 0.5 --eps 0.001 --cur_task 2 && echo "MA scenarios training Launched." 

# 4-EP
# CUDA_VISIBLE_DEVICES=7 python3 train_OUR_D.py --lr 0.001 --n_stgcnn 1 --n_txpcnn 5  --dataset Null --tag social-stgcnn-ODEP --use_lrschd --num_epochs 250 --tasks 4 --mem_size 3500 --margin 0.5 --eps 0.001 --cur_task 3 && echo "MA scenarios training Launched." 

# 5-SR
CUDA_VISIBLE_DEVICES=4 python3 train_OUR_D.py --lr 0.001 --n_stgcnn 1 --n_txpcnn 5  --dataset Null --tag social-stgcnn-ODSRT --use_lrschd --num_epochs 250 --tasks 5 --mem_size 3500 --margin 0.5 --eps 0.001 --cur_task 4 && echo "MA scenarios training Launched." 






# 1-MA
# CUDA_VISIBLE_DEVICES=7 python train_OUR.py --lr 0.001 --n_stgcnn 1 --n_txpcnn 5  --dataset 1-MA --tag social-stgcnn-O --use_lrschd --num_epochs 250 --tasks 2 --mem_size 3500 --margin 0.5 --eps 0.001 --cur_task 1 && echo "MA scenarios training Launched." 

# 2-FT
# CUDA_VISIBLE_DEVICES=6 python train_OUR.py --lr 0.001 --n_stgcnn 1 --n_txpcnn 5  --dataset 2-FT --tag social-stgcnn-OIFT --use_lrschd --num_epochs 250 --tasks 2 --mem_size 3500 --margin 0.5 --eps 0.001 --cur_task 1 && echo "FT scenarios training Launched." 

# 3-ZS
# CUDA_VISIBLE_DEVICES=5 python train_OUR.py --lr 0.001 --n_stgcnn 1 --n_txpcnn 5  --dataset 3-ZS --tag social-stgcnn-OIZS --use_lrschd --num_epochs 250 --tasks 2 --mem_size 3500 --margin 0.5 --eps 0.001 --cur_task 1 && echo "ZS scenarios training Launched." 

# 4-EP
# CUDA_VISIBLE_DEVICES=4 python train_OUR.py --lr 0.001 --n_stgcnn 1 --n_txpcnn 5  --dataset 4-EP --tag social-stgcnn-OIEP --use_lrschd --num_epochs 250 --tasks 2 --mem_size 3500 --margin 0.5 --eps 0.001 --cur_task 1 && echo "EP training Launched." 

# 5-SR
# CUDA_VISIBLE_DEVICES=3 python train_OUR.py --lr 0.001 --n_stgcnn 1 --n_txpcnn 5  --dataset 5-SR --tag social-stgcnn-OISR --use_lrschd --num_epochs 250 --tasks 2 --mem_size 3500 --margin 0.5 --eps 0.001 --cur_task 1 && echo "SR training Launched." 

# 6-JTH
# CUDA_VISIBLE_DEVICES=2 python train_OUR.py --lr 0.001 --n_stgcnn 1 --n_txpcnn 5  --dataset 6-JT --tag social-stgcnn-OIJTH --use_lrschd --num_epochs 250 --tasks 2 --mem_size 3500 --margin 0.5 --eps 0.001 --cur_task 1 && echo "JT training Launched." 

# 6-JTF
# CUDA_VISIBLE_DEVICES=1 python train_OUR.py --lr 0.001 --n_stgcnn 1 --n_txpcnn 5  --dataset 6-JT --tag social-stgcnn-OIJTF --use_lrschd --num_epochs 250 --tasks 2 --mem_size 3500 --margin 0.5 --eps 0.001 --cur_task 1 && echo "JT training Launched." 

# 7-JTH
# CUDA_VISIBLE_DEVICES=1 python train_OUR.py --lr 0.001 --n_stgcnn 1 --n_txpcnn 5  --dataset 7-JT4 --tag social-stgcnn-OIJTH4 --use_lrschd --num_epochs 250 --tasks 2 --mem_size 3500 --margin 0.5 --eps 0.001 --cur_task 1 && echo "JT training Launched." 

# 7-JTF
# CUDA_VISIBLE_DEVICES=2 python train_OUR.py --lr 0.001 --n_stgcnn 1 --n_txpcnn 5  --dataset 7-JT4 --tag social-stgcnn-OIJTF4 --use_lrschd --num_epochs 250 --tasks 2 --mem_size 3500 --margin 0.5 --eps 0.001 --cur_task 1 && echo "JT training Launched." 

# 8-JTH
# CUDA_VISIBLE_DEVICES=3 python train_OUR.py --lr 0.001 --n_stgcnn 1 --n_txpcnn 5  --dataset 8-JT3 --tag social-stgcnn-OIJTH3 --use_lrschd --num_epochs 250 --tasks 2 --mem_size 3500 --margin 0.5 --eps 0.001 --cur_task 1 && echo "JT training Launched." 

# 8-JTF
# CUDA_VISIBLE_DEVICES=4 python train_OUR.py --lr 0.001 --n_stgcnn 1 --n_txpcnn 5  --dataset 8-JT3 --tag social-stgcnn-OIJTF3 --use_lrschd --num_epochs 250 --tasks 2 --mem_size 3500 --margin 0.5 --eps 0.001 --cur_task 1 && echo "JT training Launched." 


#two continuous scenarios: MA(past)->FT(current)
# CUDA_VISIBLE_DEVICES=0 python train_GSM.py --lr 0.001 --n_stgcnn 1 --n_txpcnn 5  --dataset 2-FT --tag social-stgcnn-FT --use_lrschd --num_epochs 250 --tasks 2 --mem_size 3500 --margin 0.5 --eps 0.001 --cur_task 1 && echo "two continuous scenarios training Launched." &
# P0=$!

#three continuous scenarios: MA(past)->FT(past)->ZS(current)
#CUDA_VISIBLE_DEVICES=0 python train_GSM.py --lr 0.001 --n_stgcnn 1 --n_txpcnn 5  --dataset 3-ZS --tag social-stgcnn-ZS --use_lrschd --num_epochs 250 --tasks 3 --mem_size 3500 --margin 0.5 --eps 0.001 --cur_task 2 && echo "three continuous scenarios training Launched." &
#P1=$!

#four continuous scenarios: MA(past)->FT(past)->ZS(past)->EP(current)
#CUDA_VISIBLE_DEVICES=0 python train_GSM.py --lr 0.001 --n_stgcnn 1 --n_txpcnn 5  --dataset 4-EP --tag social-stgcnn-EP --use_lrschd --num_epochs 250 --tasks 4 --mem_size 3500 --margin 0.5 --eps 0.001 --cur_task 3 && echo "four continuous scenarios training Launched." &
#P2=$!

#five continuous scenarios: MA(past)->FT(past)->ZS(past)->EP(past)->SR(current)
#CUDA_VISIBLE_DEVICES=0 python train_GSM.py --lr 0.001 --n_stgcnn 1 --n_txpcnn 5  --dataset 5-SR --tag social-stgcnn-SR --use_lrschd --num_epochs 250 --tasks 5 --mem_size 3500 --margin 0.5 --eps 0.001 --cur_task 4 && echo "five continuous scenarios training Launched." &
#P3=$!

# wait $P0 
#$P1 $P2 $P3 
