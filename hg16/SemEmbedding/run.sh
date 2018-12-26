#!/bin/bash
cd spring2017/speech2image/SemEmbedding/

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export CUDA_HOME=/usr/local/cuda

#nohup python loaddata.py 10000 1000 captions_10k.npz images_10k.npz 1>out_loaddata.log 2>err_loaddata.log &
nohup python semanticembed.py 10000 1000 captions_10k.npz images_10k.npz 1>out_semanticembed.log 2>err_semanticembed.log &
#python scnn_test.py 5000 captions_5000.npz images_5000.npz 1 1> out_scnn_test.log 2> err_scnn_test.log &
#mv out_scnn_test.log ../data/out_scnn_tr_5000_top_1.log

#nohup python scnn_test.py 5000 captions_5000.npz images_5000.npz 5000 1> out_scnn_test.log 2> err_scnn_test.log & &&
#mv out_scnn_test.log ../data/out_scnn_tr_5000_top_5000.log

#nohup python scnn_test.py 10000 captions_10k.npz images_10k.npz 1000 1> out_scnn_test.log 2> err_scnn_test.log &
#mv out_scnn_test.log ../data/out_scnn_tr_10k_top_1k.log
#nohup python scnn_test.py 1000 captions_10k_test.npz images_10k_test.npz 1 1> out_scnn_10k_test_top_1.log 2> err_scnn_10k_test_top_1.log &

#nohup python scnn_test.py 1000 captions_10k_test.npz images_10k_test.npz 5 1> out_scnn_10k_test_top_5.log 2> err_scnn_10k_test_top_5.log &
#mv out_scnn_test.log ../data/out_scnn_10k_test_top_5.log
#nohup python scnn_test.py 128 captions_10k.npz images_10k.npz 10 1> out_scnn_10k_top_10.log 2> err_scnn_10k_top_10.log &
