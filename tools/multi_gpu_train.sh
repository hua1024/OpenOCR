export NGPUS=2
export MASTER_ADDR=127.0.0.1
export MASTER_PORT= 9001
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=$NGPUS --master_addr $MASTER_ADDR--master_port MASTER_PORT tools/train.py --distributed --config "config/det/dbnet/r50_vd_dbnet.py"