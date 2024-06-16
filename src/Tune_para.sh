# TextCNN调参
# Learning rate 0.0005, 0.005, 0.05, 0.5
for lr in 0.0005 0.005 0.05 0.5
do
    python src/TextCNN.py --learning_rate $lr
done
# Pooling size
for ps in 2 3 4 5
do
    python src/TextCNN.py --pooling_size $ps
done

# LSTM调参
# Learning rate 0.0001, 0.001, 0.01, 0.1
for lr in 0.0001 0.001 0.01 0.1
do
    python src/LSTM.py --learning_rate $lr
done
# Hidden size
for hs in 50 100 150 200
do
    python src/LSTM.py --hidden_size $hs
done
# Number of layers
for num in 1 2 3 4
do
    python src/LSTM.py --num_layers $num
done

# 初始化方法
for init in "uniform" "normal" "xavier" "orthogonal"
do
    python src/TextCNN.py --init_type $init
    python src/LSTM.py --init_type $init
done
