# cifar100 
# teacher
nohup python train_teacher.py --model resnet32x4 --gpu_id 0 --trial 0&


# student
# average
nohup python train_student.py --model_s vgg8 --teacher_num 3 --distill kd --ensemble_method AVERAGE_LOSS --nesterov -r 1 -a 1 -b 0  --trial 0 --gpu_id 4&
# FitNet-MKD
nohup python train_student.py --model_s vgg8 --teacher_num 3 --distill hint --ensemble_method AVERAGE_LOSS --nesterov -r 1 -a 1 -b 100  --trial 0 --gpu_id 3&
# EBKD
nohup python train_student.py --model_s vgg8 --teacher_num 3 --distill kd --ensemble_method EBKD --nesterov -r 1 -a 1 -b 0  --trial 0 --gpu_id 2&
# AEKD
nohup python train_student.py --model_s vgg8 --teacher_num 3 --distill kd --ensemble_method AEKD --nesterov -r 1 -a 1 -b 0 --trial 0  --gpu_id 1&
# CAMKD
nohup python train_student.py --model_s vgg8 --teacher_num 3 --distill inter --ensemble_method CAMKD --nesterov -r 1 -a 1 -b 50 --trial 0  --gpu_id 0&

