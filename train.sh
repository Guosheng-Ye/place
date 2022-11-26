#!/bin/bash
# train,infer,eval_acc,eval_fid,eval_lpips for placenet and it's imporvement model
# EXPID:experiment name
# EPOCH:max_epoch,from epoch 9 to max_epoch
# ENCODER:encoder_type for foreground,and background feature extract
EXPID=$1
epoch=$2
EPOCH=$3
ENCODER=$4

# do loop
for epoch_num in `seq ${epoch} ${EPOCH}`
do
    # main
    python main.py --data_root datasets/new_OPA/ --expid "${EXPID}_${epoch_num}" --encoder_type ${ENCODER} --n_epochs ${epoch_num}
    echo "\033[31m ${EXPID}_${epoch_num} main.py is completed!\033[0m"
    # infer for eavl
    python infer.py --data_root datasets/new_OPA/ --expid "${EXPID}_${epoch_num}" --epoch ${epoch_num} --eval_type eval --encoder_type ${ENCODER}
    echo "\033[31m ${EXPID}_${epoch_num} infer,eval_type eval is completed!\033[0m"
    # infer for evaluni repeat 10
    python infer.py --data_root datasets/new_OPA/ --expid "${EXPID}_${epoch_num}" --epoch ${epoch_num} --eval_type evaluni --repeat 10 --encoder_type ${ENCODER}
    echo "\033[31m ${EXPID}_${epoch_num} infer,eval_type evaluni is completed!\033[0m"

    # eval_acc
    cd faster-rcnn
    python generate_tsv.py --expid "${EXPID}_${epoch_num}" --epoch ${epoch_num} --eval_type "eval" --cuda
    python convert_data.py --expid "${EXPID}_${epoch_num}" --epoch ${epoch_num} --eval_type "eval"
    cd ..
    python eval/simopa_acc.py --checkpoint binary_classifier/best-acc.pth --expid "${EXPID}_${epoch_num}" --epoch ${epoch_num} --eval_type "eval"
    echo "\033[31m ${EXPID}_${epoch_num} eval_acc is completed!\033[0m"


    # eval_fid
    python eval/fid_resize299.py --expid "${EXPID}_${epoch_num}" --epoch ${epoch_num} --eval_type "eval"
    python eval/fid_score.py result/"${EXPID}_${epoch_num}"/eval/${epoch_num}/images299/ datasets/new_OPA/com_pic_testpos299/ --expid "${EXPID}_${epoch_num}" --epoch ${epoch_num} --eval_type "eval"
    echo "\033[31m ${EXPID}_${epoch_num} eval_fid is completed!\033[0m"

    # eval_lpips
    python eval/lpips_1dir.py -d result/"${EXPID}_${epoch_num}"/evaluni/${epoch_num}/images/ --expid "${EXPID}_${epoch_num}" --epoch ${epoch_num} --eval_type "evaluni" --repeat 10 --use_gpu
    echo "\033[31m ${EXPID}_${epoch_num} eval_lpips is completed!\033[0m"

done
echo "\033[31m all mession is completed!\033[0m"
