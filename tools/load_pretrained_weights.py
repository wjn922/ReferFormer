import torch

def pre_trained_model_to_finetune(checkpoint, args):
    checkpoint = checkpoint['model']
    # only delete the class_embed since the finetuned dataset has different num_classes
    num_layers = args.dec_layers + 1 if args.two_stage else args.dec_layers
    for l in range(num_layers):
        del checkpoint["class_embed.{}.weight".format(l)]
        del checkpoint["class_embed.{}.bias".format(l)]
    
    return checkpoint
