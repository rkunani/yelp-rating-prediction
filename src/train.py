"""
Fine-tunes a Model (model.py) on the Yelp dataset.
Saves model checkpoints after each epoch of training.
"""


import torch
from utils import get_dataloader, accuracy
from model import Model
from tqdm import tqdm
import numpy as np
from transformers import get_linear_schedule_with_warmup, AdamW
import time, argparse, os
import matplotlib.pyplot as plt
from itertools import cycle
import wandb

def train():

    train_dataloader, val_dataloader = get_dataloader(args.data_file, args.model_type, args.batch_size, args.num_points, args.val_size)
    val_dataloader = cycle(iter(val_dataloader))
    model = Model(args.model_type)

    # Set up W&B tracker
    if args.wandb:
      wandb.init(project="yelp-rating-prediction", notes=f"MODEL_ID: {model.MODEL_ID}")
      wandb.config.update(args)
      model.wandb_run = wandb.run.name
    
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9,0.98), weight_decay=args.weight_decay)
    
    if args.schedule =='linear':
      total_steps = len(train_dataloader) * args.epochs
      warmup_steps = total_steps * args.warmup_ratio
      scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    if args.schedule == 'plateau':
      scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience)
    
    num_updates = 0
    running_train_loss, running_train_acc = 0, 0
    running_val_loss, running_val_acc = 0, 0
    for epoch in range(1, args.epochs+1):
        print(f'Epoch {epoch} / {args.epochs}')
        
        t0 = time.time()
        model.train()
        
        for batch in tqdm(train_dataloader):

            batch = tuple(t.to(device) for t in batch)
            b_input, b_att_mask, b_labels = batch
            
            optimizer.zero_grad()
            outputs = model(b_input, attention_mask=b_att_mask, labels=b_labels)
            loss = outputs['loss']   # cross-entropy loss
            loss.backward()

            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
            optimizer.step()
            if args.schedule == 'linear':
              scheduler.step()

            # Update running quantities
            running_train_loss += loss.item()
            running_train_acc += accuracy(outputs['logits'], b_labels)
            val_batch = next(val_dataloader)
            val_batch = tuple(t.to(device) for t in batch)
            b_input, b_att_mask, b_labels = batch
            with torch.no_grad():        
                outputs = model(b_input, attention_mask=b_att_mask, labels=b_labels)
            running_val_loss += outputs['loss'].item()
            running_val_acc += accuracy(outputs['logits'], b_labels)

            # Evaluate metrics every SAVE_EVERY updates
            num_updates += 1
            if num_updates % args.save_every == 0:
              # Train
              train_loss = running_train_loss / args.save_every
              train_acc = running_train_acc / args.save_every
              print(f"Train Loss: {train_loss}")
              print(f"Train Accuracy: {train_acc}")
              if args.wandb:
                wandb.log({'train_loss': train_loss, 'train_acc': train_acc})
              running_train_loss, running_train_acc = 0, 0
              
              # Validation
              val_loss = running_val_loss / args.save_every
              val_acc = running_val_acc / args.save_every
              print(f"Val Loss: {val_loss}")
              print(f"Val Accuracy: {val_acc}")
              if args.wandb:
                wandb.log({'val_loss': val_loss, 'val_acc': val_acc})
              running_val_loss, running_val_acc = 0, 0

              # Update learning rate
              if args.schedule == 'plateau':
                scheduler.step(val_loss)
                if args.wandb:
                  wandb.log({'learning_rate': scheduler.optimizer.param_groups[0]['lr']})

        print(f"Training epoch took: {time.time() - t0}s")

        # Checkpoint
        CKPT_DIR = f"{args.checkpoint_dir}/{model.MODEL_ID}"
        if not os.path.exists(CKPT_DIR):
            os.makedirs(CKPT_DIR)
        CKPT_FILE = CKPT_DIR + f"/epoch{epoch}.pkl"
        torch.save(model.state_dict(), CKPT_FILE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="distilbert-base-cased")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num_points", type=int, default=10)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument('--schedule', type=str, default='linear')
    parser.add_argument('--factor', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--warmup_ratio', type=float, default=0.06)
    parser.add_argument('--grad_clip', type=float, default=0.0)
    parser.add_argument('--wandb', action="store_true", default=False)
    parser.add_argument("--checkpoint_dir", type=str, default="../checkpoint")
    parser.add_argument("--data_file", type=str, default="../yelp_dataset/review_sample.json")
    parser.add_argument("--save_every", type=int, default=100)

    args = parser.parse_args()
    # Use GPU if available
    if torch.cuda.is_available():        
        device = torch.device("cuda")
        print('Use GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using CPU instead.')
        device = torch.device("cpu")

    train()