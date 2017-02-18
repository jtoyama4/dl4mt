# MVNMT 

## Training
1. `cd <modeldir>`
2. `THEANO_FLAGS="device=gpu0" bash train.sh`

## Eval
1. `cd <modeldir>`
2. `bash translate.sh` or `bash translate_all.sh`
3. `bash eval_meteor.sh` or `bash eval_result_all.sh`

## plot
1. `cd <modeldir>`
2. `bash plot.sh`
3. see `result/{cost,valcost}.png`

## modeldir
- `session3` (nmt)
- `mvnmt2`
- `mvnmt2_att`
- `mvnmt2_add_noise`

## hyperparameters (common to all)
- `dim`: 256
- `dim_word`: 256
- `dimv`: 256
- `dim_pic`: 1024
- `decay_c`: 0.001
- `lr`: 1.0
- `batchsize`: 32
- `finetunedir`: "nmt"
