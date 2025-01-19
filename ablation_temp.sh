CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/multi_queue_usl.py -b 256 -a resnet50 -d market1501 --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 --lr 0.00035 --epochs 50
 CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/multi_queue_infomap.py -b 256 -a resnet50 -d market1501 --iters 200 --momentum 0.1 --eps 0.5 --k1 15 --k2 4 --num-instances 16 --lr 0.00035 --epochs 50

CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/multi_queue_usl.py -b 256 -a resnet50 -d msmt17 --iters 400 --momentum 0.1 --eps 0.6 --num-instances 16 --lr 0.00035 --epochs 50
 CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/multi_queue_infomap.py -b 256 -a resnet50 -d msmt17 --iters 400 --momentum 0.1 --eps 0.5 --k1 15 --k2 4 --num-instances 16 --lr 0.00035 --epochs 50

CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/multi_queue_usl.py -b 256 -a resnet50 -d dukemtmcreid --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 --lr 0.00035 --epochs 50
 CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/multi_queue_infomap.py -b 256 -a resnet50 -d dukemtmcreid --iters 200 --momentum 0.1 --eps 0.5 --k1 15 --k2 4 --num-instances 16 --lr 0.00035 --epochs 50

CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/multi_queue_usl.py -b 256 -a resnet50 -d personx --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 --lr 0.00035 --epochs 50
 CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/multi_queue_infomap.py -b 256 -a resnet50 -d personx --iters 200 --momentum 0.1 --eps 0.5 --k1 15 --k2 4 --num-instances 16 --lr 0.00035 --epochs 50

CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/multi_queue_usl.py -b 256 -a resnet50 -d veri --iters 400 --momentum 0.1 --eps 0.6 --num-instances 16 --height 224 --width 224
 CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/multi_queue_infomap.py -b 256 -a resnet50 -d veri --iters 400 --momentum 0.1 --eps 0.5 --k1 15 --k2 4 --num-instances 16 --height 224 --width 224
