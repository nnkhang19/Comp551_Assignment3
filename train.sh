
#!/bin/bash

seed=42
lr=1e-2
num_epochs=20
batch_size=512
num_workers=4

# python3 main.py --seed $seed --lr $lr --num_epochs $num_epochs --batch_size $batch_size \
#                 --num_workers $num_workers \
#                 --exp_name unnormalized --normalized 0 \
#                 --hidden_dims 256 256 --acts relu relu

python3 main.py --seed $seed --lr $lr --num_epochs $num_epochs --batch_size $batch_size \
                --num_workers $num_workers \
                --exp_name img_size_128 --normalized 1 --img_size 128 \
                --hidden_dims 256 256 --acts relu relu

# python3 main.py --seed $seed --lr $lr --num_epochs $num_epochs --batch_size $batch_size \
#                 --num_workers $num_workers  --reg_type 'l2' --reg_coef 0.001 \
#                 --exp_name l2_reg_1e3 \
#                 --hidden_dims 256 256 --acts relu relu

# python3 main.py --seed $seed --lr $lr --num_epochs $num_epochs --batch_size $batch_size \
#                 --num_workers $num_workers --reg_type 'l2' --reg_coef 0.0001 \
#                 --exp_name l2_reg_1e4 \
#                 --hidden_dims 256 256 --acts relu relu

# python3 main.py --seed $seed --lr $lr --num_epochs $num_epochs --batch_size $batch_size \
#                 --num_workers $num_workers  --reg_type 'l2' --reg_coef 0.00001 
#                 --exp_name l2_reg_1e5 \
#                 --hidden_dims 256 256 --acts relu relu