BATCH_SIZE=64
EPOCHS=100
INITIAL_LR=0.001
MIN_LR=0.00001
PATIENCE=5
MIN_DELTA=0.1
VERBOSE=1
NET_TYPE=1

FILTERS=16
KERNEL_SIZE=21

echo " "
python MAIN.py --batch_size $BATCH_SIZE --epochs $EPOCHS --filters $FILTERS \
	--kernel_size $KERNEL_SIZE --initial_lr $INITIAL_LR --min_lr $MIN_LR \
	--patience $PATIENCE --min_delta $MIN_DELTA --verbose $VERBOSE --net_type $NET_TYPE


