# How ro run?
## Enviroment setup
### Conda
conda env create -f  env/IKENet.yml

### Run code
python train.py -i #INPUT_PATH -o #OUTPUT_PATH --z_up ZSSR --INN --test --tri_channel --use_res --linear --other_loss interpolation_loss cycle_consistency --scale_factor #SCALE_FACTOR(0.5/0.25) --INN_down space2depth --tb_name #TENSORBOARD_PATH