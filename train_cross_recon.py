import argparse
parser = argparse.ArgumentParser(description='Concept-level Representation.')
parser.add_argument('--dataset', default='imagenet', type=str, help='imagenet, cub')
parser.add_argument('--model', default='ViT-L-14', type=str, help='ViT-B-32, ViT-L-14')
parser.add_argument('--pretrained', default='laion2b_s32b_b82k', type=str, help='laion2b_s32b_b82k, openai')
parser.add_argument('--batch_size', default=256, type=int, help='mini-batch size (default: 256)')
parser.add_argument('--seed', default=0, type=int, help='random seed (default: 0)')
parser.add_argument('--gpu_id', default="0", type=str, help='gpu id')
parser.add_argument('--knowledge_path', default="./", type=str, help='path to parsed knowledge graph.')
parser.add_argument('--save_path', default="./", type=str, help='path to folder saving the checkpoints.')
parser.add_argument('--lambda_recon_image', default=0.01, type=float, help='Lambda Image Reconstruction.')
parser.add_argument('--lambda_recon_text', default=0.005, type=float, help='Lambda Text Reconstruction.')
parser.add_argument('--lambda_orth', default=0.01, type=float, help='Lambda Orthogonal.')
parser.add_argument('--lambda_spa', default=0.001, type=float, help='Lambda Sparcity.')
parser.add_argument('--save_freq', default=5, type=int, help='saving frequency (steps).')
parser.add_argument('--checkpoint', default=None, type=str, help='path to the checkpoint.')
parser.add_argument('--freeze_text', action='store_true', help='freeze clip text encoder or not.')
parser.add_argument('--max_step', default=50000, type=int, help='maximum training steps.')
parser.add_argument('--epochs', default=20, type=int, help='maximum training epochs.')
parser.add_argument('--lr', default=5e-5, type=float, help='learning rate.')
parser.add_argument('--eval', action='store_true', help='freeze batch norm.')
parser.add_argument('--T_max', default=100, type=int, help='T-max for CosineAnnealingLR.')
parser.add_argument('--start_step', default=0, type=int, help='T-max for CosineAnnealingLR.')
args = parser.parse_args()
print(args)


import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

import numpy as np
import torch
from PIL import Image
import os.path
import argparse
from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.factory import create_model_and_transforms, get_tokenizer
from utils.binary_waterbirds import BinaryWaterbirds
from prs_hook import hook_prs_logger
from torchvision.datasets import CIFAR100, CIFAR10, ImageNet, ImageFolder
from torch.nn import functional as F

import torch.nn as nn
from load import *
import random
from torch import optim
from loss import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import *


def zeroshot_classifier(classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            texts = tokenizer(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


hparams['seed'] = args.seed
hparams['batch_size'] = args.batch_size
hparams['model_size'] = args.model

seed_everything(hparams['seed'])

bs = hparams['batch_size']
dataloader = DataLoader(dataset, bs, shuffle=True, num_workers=16, pin_memory=True)


# load model
device = torch.device(hparams['device'])
model, _, preprocess = create_model_and_transforms(
    args.model, pretrained=args.pretrained
)
model.to(device)
# model.eval()
context_length = model.context_length
vocab_size = model.vocab_size

print(
    "Model parameters:",
    f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}",
)
print("Context length:", context_length)
print("Vocab size:", vocab_size)
print("Len of res:", len(model.visual.transformer.resblocks))

prs = hook_prs_logger(model, device)
if args.checkpoint != None:
    print("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict']) 

if args.eval:
    model.eval()
else:
    model.train()

if args.freeze_text:
    for param in model.transformer.parameters():
        param.requires_grad = False
model = model.to(device)

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
loss_orth = BatchOrthogonalLoss()
loss_spa = SparsityLoss()

optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9,0.999),eps=1e-8, weight_decay=0.001) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
scheduler = CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=0)
tokenizer = get_tokenizer(args.model)

with open(args.knowledge_path, 'r') as file:
    description_relation = json.load(file)

def flatten_and_store(source_list, target_list, indices_key):
    start_index = len(target_list)
    target_list.extend(source_list)
    end_index = len(target_list) - 1
    indices[indices_key].append((start_index, end_index))


step = 0
step_ctr = 0.0
step_recon_image = 0.0
step_recon_text = 0.0
step_orth = 0.0
step_spa = 0.0
step_total = 0.0

for epoch in range(args.epochs):
    if step > args.max_step:
            print("Max step, training terminated!")
            break

    for batch_number, batch in enumerate(tqdm(dataloader)):
        if step > args.max_step:
            print("Max step, training terminated!")
            break

        if step <= args.start_step:
            step += 1
            optimizer.step()
            scheduler.step()
            continue

        optimizer.zero_grad()

        if (step-1) % args.save_freq == 0:
            step_ctr = 0.0
            step_recon_image = 0.0
            step_recon_text = 0.0
            step_orth = 0.0
            step_spa = 0.0
            step_total = 0.0
            # count = 0

        weights = torch.tensor([0.1, 0.2, 0.7]).to(device)
        logit_scale = model.logit_scale.exp()
        images, labels = batch
        images = images.to(device)

        texts = np.array(label_to_classname)[labels].tolist()

        tokenized_concepts_list = []
        rich_labels = []
        for i in range(len(texts)):
            concepts = gpt_descriptions[texts[i]][:5]
            concatenated_concepts = ', '.join(concepts)
            label = hparams['label_before_text'] + wordify(texts[i]) + hparams['label_after_text'] + " It may contains " + concatenated_concepts
            rich_labels.append(label)
            
            concepts.insert(0, texts[i])
            tokenized_concepts = tokenizer(concepts)
            tokenized_concepts_list.append(tokenized_concepts)
        
        images = images.to(device)
        # if args.augment_text:
        rich_labels = tokenizer(rich_labels)
        texts = rich_labels.to(device)
        
        prs.reinit()
        representation = model.encode_image(
            images, attn_method="head", normalize=False
        )
        attentions = prs.finalize(representation)

        description_embeddings = model.encode_text(texts)

        tokenized_concepts_list = torch.stack(tokenized_concepts_list).reshape(-1, 77)
        node_text_embeddings = model.encode_text(tokenized_concepts_list.to(device))
        node_text_embeddings = node_text_embeddings.reshape(len(images), -1, 512)

        attentions_maps = []
        for i in range(len(attentions)):
            # attentions_map = attentions[i, :, 1:].sum(axis=(0)) @ node_text_embeddings[i].T
            attentions_map = (attentions[i, -3:, 1:, :] * weights[:, None, None, None]).sum(axis=(0, 2)) @ node_text_embeddings[i].T
            attentions_maps.append(attentions_map.permute(1, 0))
        attentions_maps = torch.stack(attentions_maps)
    
        image_tokens = attentions[:, :, 1:].sum(axis=(1,3))
        node_image_representations = []
        for i in range(len(image_tokens)):
            node_image_representation = []
            for j in range(len(attentions_maps[i])):
                node_image_representation.append(image_tokens[i][attentions_maps[i][j] > attentions_maps[i][j].mean()].mean(dim=0))
            node_image_representations.append(torch.stack(node_image_representation))
        node_image_representations = torch.stack(node_image_representations)

        recon_image_embeddings = []
        for i in range(len(node_image_representations)):
            recon_image_embeddings.append(node_image_representations[i][1:].sum(axis=(0)))
        recon_image_embeddings = torch.stack(recon_image_embeddings)

        category_text_embeddings = []
        # recon_text_embeddings = []
        for i in range(len(node_text_embeddings)):
            category_text_embeddings.append(node_text_embeddings[i][0])
            # recon_text_embeddings.append(node_text_embeddings[i][1:].sum(axis=(0)))
        category_text_embeddings = torch.stack(category_text_embeddings)
        # recon_text_embeddings = torch.stack(recon_text_embeddings)

        ground_truth_recon_image = torch.arange(len(images)).to(device)
        ground_truth_recon_text = torch.arange(len(images)).to(device)
        logit_scale_recon_image = model.logit_scale.exp()
        logit_scale_recon_text = model.logit_scale.exp()

        logits_per_image_recon_image, logits_per_text_recon_image = create_logits(recon_image_embeddings, category_text_embeddings, logit_scale_recon_image)
        recon_image_loss = (loss_img(logits_per_image_recon_image, ground_truth_recon_image) + loss_txt(logits_per_text_recon_image, ground_truth_recon_image))/2
        # logits_per_image_recon_text, logits_per_text_recon_text = create_logits(representation, recon_text_embeddings, logit_scale_recon_text)
        # recon_text_loss = (loss_img(logits_per_image_recon_text, ground_truth_recon_text) + loss_txt(logits_per_text_recon_text, ground_truth_recon_text))/2

        orth_loss = loss_orth(attentions_maps)
        spa_loss = loss_spa(attentions_maps)
        
        ground_truth = torch.arange(len(images)).to(device)
        logits_per_image, logits_per_text = create_logits(representation, description_embeddings, logit_scale)
        clip_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
        step_ctr += clip_loss.item()

        batch_recon_image = args.lambda_recon_image * recon_image_loss
        # batch_recon_text = args.lambda_recon_text * recon_text_loss
        batch_orth = args.lambda_orth * orth_loss
        batch_spa = args.lambda_spa * spa_loss
        step_recon_image += batch_recon_image.item()
        # step_recon_text += batch_recon_text.item()
        step_orth += batch_orth.item()
        step_spa += batch_spa.item()

        total_loss = clip_loss + batch_recon_image + batch_orth + batch_spa
        step_total += total_loss.item()
        total_loss.backward()

        optimizer.step()
        scheduler.step()

        # count += 1

        if step % args.save_freq == 0 and step > args.start_step:
            print("\n" + "Saving logs...")

            with open(f"{args.save_path}/logs/ctr_loss.log", "a") as f:
                f.write(str(step) + ' ' + str(step_ctr/args.save_freq) + '\n')
            with open(f"{args.save_path}/logs/recon_image_loss.log", "a") as f:
                f.write(str(step) + ' ' + str(step_recon_image/args.save_freq) + '\n')
            with open(f"{args.save_path}/logs/orth_loss.log", "a") as f:
                f.write(str(step) + ' ' + str(step_orth/args.save_freq) + '\n')
            with open(f"{args.save_path}/logs/spa_loss.log", "a") as f:
                f.write(str(step) + ' ' + str(step_spa/args.save_freq) + '\n')
            with open(f"{args.save_path}/logs/total_loss.log", "a") as f:
                f.write(str(step) + ' ' + str(step_total/args.save_freq) + '\n')

            print("Saving checkpoint...")
            torch.save({
                'epoch': step,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss,
            }, f"{args.save_path}/step_{step}.pt") #just change to your preferred folder/filename

            
        step += 1