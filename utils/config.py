ROOT=r'/data01/yuho_hdd/PromptCAL'
DATA_ROOT=r'/home/miil/Datasets/'

# -----------------
# DATASET PATHS
# -----------------
cifar_10_root = f'{DATA_ROOT}/FSCIL-CEC'
cifar_100_root = f'{DATA_ROOT}/FSCIL-CEC'
cub_root = f'{DATA_ROOT}/FSCIL-CEC'
aircraft_root = f'{DATA_ROOT}/GCD/fcvc_aircraft'
herbarium_dataroot = f'{DATA_ROOT}/GCD/herbarium_19'
imagenet_root = f'{DATA_ROOT}/GCD/imagenet'
imagenet_gcd_root = f'{DATA_ROOT}/imagenet_100_gcd'

# -----------------
# OTHER PATHS
# -----------------
osr_split_dir = f'{ROOT}/data/ssb_splits' # OSR Split dir
feature_extract_dir = f'{ROOT}/tmp'     # Extract features to this directory
exp_root = f'{ROOT}/cache'          # All logs and checkpoints will be saved here

# -----------------
# OTHER PATHS
# -----------------
dino_pretrain_path = r'/data01/yuho_hdd/refactored_gcd/cache/pretrained_model/dino_vitbase16_pretrain.pth'
ibot_pretrain_path = r'/home/sheng/dino/checkpoint/ibot-b16t.pth'   # deprecated


