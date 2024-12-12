nohup python train_cross_recon.py \
--dataset imagenet \
--model ViT-B-32 \
--pretrained "openai" \
--batch_size 256 \
--seed 0 \
--gpu_id "4" \
--knowledge_path "../Knowledge Graphs/imagenet_description_relation.json" \
--save_path "" \
--lambda_recon_image 0.5 \
--lambda_recon_text 0.5 \
--lambda_orth 0.001 \
--lambda_spa 0.0 \
--save_freq 2000 \
--max_step 17000 \
--lr 5e-7 \
--eval \
--T_max 17000 \
--start_step 0 \
> "run.log" 2>&1 &