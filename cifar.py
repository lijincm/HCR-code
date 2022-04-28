import os
the_command = 'python main.py' \
        + ' --n_epochs 50'   \
        + ' --lr 0.3' \
        + ' --minibatch_size 32' \
        + ' --batch_size 32' \
        + ' --alpha 0.5' \
        + ' --beta 0.5' \
        + ' --gamma 0.5' \
        + ' --csv_log'   \
        + ' --tensorboard'   \
        + ' --model hcr' \
        + ' --buffer_size 200' \
        + ' --dataset seq-cifar10'\
        + ' --seed 60' \
        + ' --sigmoid 0.3' \
        
        

os.system(the_command)








    





