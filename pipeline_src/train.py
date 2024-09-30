import torch
import os
import numpy as np
from sacrebleu.metrics import BLEU
from .trainer.train_epoch import train_epoch, validate, predict, save_model



def train(
    model,
    tokenizer,
    train_loaders,
    val_loader,
    test_loader,
    optimizer,
    scheduler,
    logger,
    accelerator,
    config,
):

    for epoch in range(config.n_epochs):
        print(f"Start of the epoch {epoch}")
        
        train_epoch(
            model,
            tokenizer,
            optimizer,
            scheduler,
            train_loaders,
            val_loader,
            logger,
            accelerator,
            epoch=epoch,
            config=config
        )

        if (epoch + 1) % config.validation == 0:
            validate(model, val_loader, logger, accelerator, config)
            if accelerator.is_main_process:
                print("validated")

        if ((epoch + 1) % config.save_every == 0) and (config.save_strategy == 'epoch'):
            save_model(model, tokenizer, config, accelerator=accelerator, epoch=epoch)

        if (epoch + 1) % config.compute_metrics_every == 0:
            all_preds, all_labels = predict(
                model, tokenizer, test_loader, config, accelerator=accelerator, epoch=epoch
            )

            spbleu = BLEU(tokenize='flores101').corpus_score(all_preds, [all_labels]).score
            bleu = BLEU().corpus_score(all_preds, [all_labels]).score
            metrics = {
                'spbleu': spbleu,
                'bleu': bleu
            }
            for key in metrics:
                logger.add_scalar(key, float(metrics[key]))

        st = logger.get_step()
        if st >= config.max_steps:
            break

