cd ../../

echo
echo -n "* Please select GPU ID: "
read -r num
GPU="${num}"

SAVE_DIR=/data01/yuho_hdd/PromptCAL/cache
mkdir -p ${SAVE_DIR}/kmeans

### ==============================================================================================

python -m methods.clustering.extract_features \
  --dataset scars \
  --warmup_model_dir '/data01/yuho_hdd/refactored_gcd/cache/promptcal/stage1/scars/model_best.pt' \
  --model_name 'vpt-model' \
  --transform 'imagenet' \
  --num_prompts 5 \
  --device "cuda:${GPU}" \
  --n_shallow_prompts 0 \
  --with_parallel 'False' \
  --save_name stage_1

python -m methods.clustering.k_means \
  --dataset scars \
  --semi_sup 'True' \
  --use_ssb_splits 'True' \
  --max_kmeans_iter 200 \
  --k_means_init 100 \
  --model_name 'vpt-model' \
  --device "cuda:${GPU}" \
  --eval_test 'False' \
  --save_name stage_1 \
  >> ${SAVE_DIR}/kmeans/stage_1_scars.out

python -m methods.clustering.extract_features \
  --dataset cifar100 \
  --warmup_model_dir '/data01/yuho_hdd/refactored_gcd/cache/promptcal/stage1/cifar100/model_best.pt' \
  --model_name 'vpt-model' \
  --transform 'imagenet' \
  --num_prompts 5 \
  --device "cuda:${GPU}" \
  --n_shallow_prompts 0 \
  --with_parallel 'False' \
  --save_name stage_1

python -m methods.clustering.k_means \
  --dataset cifar100 \
  --semi_sup 'True' \
  --use_ssb_splits 'True' \
  --max_kmeans_iter 200 \
  --k_means_init 100 \
  --model_name 'vpt-model' \
  --device "cuda:${GPU}" \
  --eval_test 'False' \
  --save_name stage_1 \
  >> ${SAVE_DIR}/kmeans/stage_1_cifar100.out

python -m methods.clustering.extract_features \
  --dataset cub \
  --warmup_model_dir '/data01/yuho_hdd/refactored_gcd/cache/promptcal/stage1/cub/model_best.pt' \
  --model_name 'vpt-model' \
  --transform 'imagenet' \
  --num_prompts 5 \
  --device "cuda:${GPU}" \
  --n_shallow_prompts 0 \
  --with_parallel 'False' \
  --save_name stage_1

python -m methods.clustering.k_means \
  --dataset cub \
  --semi_sup 'True' \
  --use_ssb_splits 'True' \
  --max_kmeans_iter 200 \
  --k_means_init 100 \
  --model_name 'vpt-model' \
  --device "cuda:${GPU}" \
  --eval_test 'False' \
  --save_name stage_1 \
  >> ${SAVE_DIR}/kmeans/stage_1_cub1.out

python -m methods.clustering.k_means \
  --dataset cub \
  --semi_sup 'True' \
  --use_ssb_splits 'True' \
  --max_kmeans_iter 200 \
  --k_means_init 200 \
  --model_name 'vpt-model' \
  --device "cuda:${GPU}" \
  --eval_test 'False' \
  --save_name stage_1 \
  >> ${SAVE_DIR}/kmeans/stage_1_cub2.out
