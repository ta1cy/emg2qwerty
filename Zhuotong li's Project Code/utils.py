from torch.nn.utils.rnn import pad_sequence
import torch


def collate_fn(batch):

    xs = [b[0] for b in batch]
    ys = [b[1] for b in batch]

    x_lengths = torch.tensor([len(x) for x in xs])
    y_lengths = torch.tensor([len(y) for y in ys])

    xs = pad_sequence(xs, batch_first=True)

    ys = torch.cat(ys)

    return xs, x_lengths, ys, y_lengths


def decode_tokens(tokens, idx2char, blank_id=0):

    chars = []

    for token in tokens:
        if token == blank_id:
            continue
        if token in idx2char:
            chars.append(idx2char[token])

    return "".join(chars)


def levenshtein(a, b):

    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]

    for i in range(len(a) + 1):
        dp[i][0] = i

    for j in range(len(b) + 1):
        dp[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    return dp[-1][-1]


def batch_cer_stats(pred_tokens, targets, target_lengths, idx2char, blank_id=0):

    total_edits = 0
    total_chars = 0
    offset = 0

    target_lengths_list = target_lengths.tolist()
    for i, target_len in enumerate(target_lengths_list):
        gt_tokens = targets[offset:offset + target_len].tolist()
        offset += target_len

        pred_text = decode_tokens(pred_tokens[i], idx2char, blank_id=blank_id)
        gt_text = decode_tokens(gt_tokens, idx2char, blank_id=blank_id)

        total_edits += levenshtein(pred_text, gt_text)
        total_chars += max(1, len(gt_text))

    return total_edits, total_chars
