import argparse
import warnings

import torch
from torch.utils.data import DataLoader

from dataset import EMGDataset
from model import build_model, ctc_beam_decode, ctc_greedy_decode
from split_utils import resolve_split_files
from tokenizer import Tokenizer
from utils import batch_cer_stats, collate_fn

warnings.filterwarnings(
    'ignore',
    message='enable_nested_tensor is True, but self.use_nested_tensor is False',
)


def evaluate(
    model,
    loader,
    device,
    blank_id,
    idx2char,
    decode_mode='greedy',
    beam_size=10,
    beam_token_prune=10,
):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    total_edits = 0
    total_chars = 0

    with torch.no_grad():
        for x, x_lengths, targets, target_lengths in loader:
            x = x.to(device)
            x_lengths = x_lengths.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            log_probs, out_lengths = model(x, x_lengths)
            loss = model.loss(log_probs, out_lengths, targets, target_lengths)

            if decode_mode == 'beam':
                pred_tokens = ctc_beam_decode(
                    log_probs,
                    out_lengths,
                    blank_id=blank_id,
                    beam_size=beam_size,
                    token_prune=beam_token_prune,
                )
            else:
                pred_tokens = ctc_greedy_decode(log_probs, out_lengths, blank_id=blank_id)
            edits, chars = batch_cer_stats(
                pred_tokens=pred_tokens,
                targets=targets.detach().cpu(),
                target_lengths=target_lengths.detach().cpu(),
                idx2char=idx2char,
                blank_id=blank_id,
            )

            total_loss += float(loss.item())
            total_batches += 1
            total_edits += edits
            total_chars += chars

    avg_loss = total_loss / max(1, total_batches)
    cer = total_edits / max(1, total_chars)
    return avg_loss, cer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='89335547')
    parser.add_argument('--window', type=int, default=2000)
    parser.add_argument('--train-stride', type=int, default=500)
    parser.add_argument('--eval-stride', type=int, default=2000)
    parser.add_argument('--context-left', type=int, default=400)
    parser.add_argument('--context-right', type=int, default=400)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--noise-std', type=float, default=0.01)
    parser.add_argument('--channel-dropout', type=float, default=0.05)
    parser.add_argument('--val-decode', type=str, default='beam', choices=['greedy', 'beam'])
    parser.add_argument('--beam-size', type=int, default=10)
    parser.add_argument('--beam-token-prune', type=int, default=10)
    parser.add_argument(
        '--model-type',
        type=str,
        default='transformer',
        choices=['transformer', 'cnn_transformer'],
    )
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--dim-feedforward', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--cnn-layers', type=int, default=4)
    parser.add_argument('--cnn-kernel-size', type=int, default=5)
    parser.add_argument('--ckpt-best', type=str, default='ckpt_best.pt')
    parser.add_argument('--ckpt-last', type=str, default='ckpt_last.pt')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = Tokenizer()

    train_files = resolve_split_files(args.data_dir, 'train')
    val_files = resolve_split_files(args.data_dir, 'val')
    test_files = resolve_split_files(args.data_dir, 'test')

    print(
        f"Using official split: train={len(train_files)} "
        f"val={len(val_files)} test={len(test_files)} sessions"
    )

    print('Computing train-set per-channel mean/std...')
    norm_mean, norm_std = EMGDataset.compute_channel_stats(train_files)
    print('Computed normalization stats.')

    train_dataset = EMGDataset(
        args.data_dir,
        tokenizer,
        window=args.window,
        stride=args.train_stride,
        context_left=args.context_left,
        context_right=args.context_right,
        files=train_files,
        norm_mean=norm_mean,
        norm_std=norm_std,
        augment=True,
        noise_std=args.noise_std,
        channel_dropout_prob=args.channel_dropout,
    )
    val_dataset = EMGDataset(
        args.data_dir,
        tokenizer,
        window=args.window,
        stride=args.eval_stride,
        context_left=args.context_left,
        context_right=args.context_right,
        files=val_files,
        norm_mean=norm_mean,
        norm_std=norm_std,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    num_classes = len(tokenizer.char2idx)
    blank_id = 0

    model_kwargs = {
        'd_model': args.d_model,
        'nhead': args.nhead,
        'num_layers': args.num_layers,
        'dim_feedforward': args.dim_feedforward,
        'dropout': args.dropout,
    }
    if args.model_type == 'cnn_transformer':
        model_kwargs['cnn_layers'] = args.cnn_layers
        model_kwargs['cnn_kernel_size'] = args.cnn_kernel_size

    print(f"Model type: {args.model_type} | model_kwargs={model_kwargs}")

    model = build_model(
        num_classes=num_classes,
        blank_id=blank_id,
        model_type=args.model_type,
        model_kwargs=model_kwargs,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_cer = float('inf')
    no_improve_epochs = 0

    for epoch in range(args.epochs):
        model.train()
        running_train_loss = 0.0
        train_batches = 0

        for x, x_lengths, targets, target_lengths in train_loader:
            x = x.to(device)
            x_lengths = x_lengths.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            log_probs, out_lengths = model(x, x_lengths)
            loss = model.loss(log_probs, out_lengths, targets, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_train_loss += float(loss.item())
            train_batches += 1

        train_loss = running_train_loss / max(1, train_batches)
        val_loss, val_cer = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            blank_id=blank_id,
            idx2char=tokenizer.idx2char,
            decode_mode=args.val_decode,
            beam_size=args.beam_size,
            beam_token_prune=args.beam_token_prune,
        )

        payload = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'norm_mean': norm_mean.tolist(),
            'norm_std': norm_std.tolist(),
            'epoch': epoch + 1,
            'val_cer': float(val_cer),
            'window': args.window,
            'context_left': args.context_left,
            'context_right': args.context_right,
            'val_decode': args.val_decode,
            'beam_size': args.beam_size,
            'beam_token_prune': args.beam_token_prune,
            'model_type': args.model_type,
            'model_kwargs': model_kwargs,
        }
        torch.save(payload, args.ckpt_last)

        improved = val_cer < best_val_cer
        if improved:
            best_val_cer = val_cer
            no_improve_epochs = 0
            torch.save(payload, args.ckpt_best)
        else:
            no_improve_epochs += 1

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_CER={val_cer:.4f} "
            f"best_val_CER={best_val_cer:.4f}"
        )

        if no_improve_epochs >= args.patience:
            print(
                f"Early stopping triggered after {args.patience} "
                f"epochs without val CER improvement."
            )
            break

    print(f"Saved last checkpoint to {args.ckpt_last}")
    print(f"Saved best checkpoint to {args.ckpt_best}")


if __name__ == '__main__':
    main()
