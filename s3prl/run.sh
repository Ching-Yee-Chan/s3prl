# ASR
UDA_VISIBLE_DEVICES=2 python3 run_downstream.py -n EnCodec -m train -u encodec -d asr
# test without LM
CUDA_VISIBLE_DEVICES=3 python3 run_downstream.py -m evaluate -t "test-clean" -e result/downstream/EnCodec/dev-clean-best.ckpt
# test with LM
CUDA_VISIBLE_DEVICES=3 python3 run_downstream.py -m evaluate -t "test-clean" -e /mnt/users/hccl.local/jkzhao/projects/s3prl/s3prl/result/downstream/EnCodec/dev-clean-best.ckpt \
    -o "\
        config.downstream_expert.datarc.decoder_args.decoder_type='kenlm',, \
        config.downstream_expert.datarc.decoder_args.kenlm_model='/mnt/users/hccl.local/jkzhao/ckpts/4-gram.arpa.gz',, \
        config.downstream_expert.datarc.decoder_args.lexicon='/mnt/users/hccl.local/jkzhao/ckpts/librispeech_lexicon.lst' \
       "

# IC
CUDA_VISIBLE_DEVICES=3 python3 run_downstream.py -n EnCodec_ic -m train -u encodec -d fluent_commands
CUDA_VISIBLE_DEVICES=1 python3 run_downstream.py -m evaluate -e result/downstream/EnCodec_ic/dev-best.ckpt

# SI
CUDA_VISIBLE_DEVICES=1 python3 run_downstream.py -n EnCodec_si -m train -u encodec -d voxceleb1
CUDA_VISIBLE_DEVICES=0 python3 run_downstream.py -m evaluate -e result/downstream/EnCodec_si/dev-best.ckpt
