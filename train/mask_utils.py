import torch


def get_standard_grid_mask(batch_size0, batch_size1, pred_step, last_size, device="cuda"):
    B0, B1, N, LS = batch_size0, batch_size1, pred_step, last_size
    device = torch.device(device)

    assert B0 <= B1, "Invalid B0, B1: {} {}".format(B0, B1)

    # mask meaning: -2: omit, -1: temporal neg (hard), 0: easy neg, 1: pos, -3: spatial neg
    mask = torch.zeros((B0, N, LS ** 2, B1, N, LS ** 2), dtype=torch.int8, requires_grad=False).detach().to(device)
    # spatial neg pairs
    mask[torch.arange(B0), :, :, torch.arange(B0), :, :] = -3
    # temporal neg pairs
    for k in range(B0):
        mask[k, :, torch.arange(LS ** 2), k, :, torch.arange(LS ** 2)] = -1
    tmp = mask.permute(0, 2, 1, 3, 5, 4).contiguous().view(B0 * LS ** 2, N, B1 * LS ** 2, N)
    # positive pairs
    for j in range(B0 * LS ** 2):
        tmp[j, torch.arange(N), j, torch.arange(N - N, N)] = 1
    mask = tmp.view(B0, LS ** 2, N, B1, LS ** 2, N).permute(0, 2, 1, 3, 5, 4)
    # Final shape: (B, N, LS**2, B, N, LS**2)
    assert torch.allclose(mask[:, :, :, B0:, :, :], torch.tensor(0, dtype=torch.int8)), "Invalid values"

    return mask


def get_multi_modal_grid_mask(batch_size0, batch_size1, pred_step, last_size0, last_size1, device="cuda"):
    B0, B1, N, LS0, LS1 = batch_size0, batch_size1, pred_step, last_size0, last_size1
    device = torch.device(device)

    assert B0 <= B1, "Invalid B0, B1: {} {}".format(B0, B1)

    # mask meaning: -2: omit, -1: temporal neg (hard), 0: easy neg, 1: pos, -3: spatial neg
    mask = torch.zeros((B0, N, LS0 ** 2, B1, N, LS1 ** 2), dtype=torch.int8, requires_grad=False).detach().to(device)
    # spatial neg pairs
    mask[torch.arange(B0), :, :, torch.arange(B0), :, :] = -3

    # temporal neg pairs
    for k in range(B0):
        mask[k, :, torch.arange(LS0 ** 2), k, :, torch.arange(LS1 ** 2)] = -1
    tmp = mask.permute(0, 2, 1, 3, 5, 4).contiguous().view(B0, LS0, LS0, N, B1, LS1, LS1, N)
    # shape: (B, LS0, LS0, N, B, LS1, LS1, N)

    # Generate downsamplings
    ds0, ds1 = LS0 // min(LS0, LS1), LS1 // min(LS0, LS1)

    # positive pairs
    for j in range(B0):
        for i in range(min(LS0, LS1)):
            tmp[j, i * ds0:(i + 1) * ds0, i * ds0:(i + 1) * ds0, torch.arange(N),
            j, i * ds1:(i + 1) * ds1, i * ds1:(i + 1) * ds1, torch.arange(N)] = 1

    # Sanity check
    for ib in range(B0):
        for jn in range(N):
            for jls0 in range(LS0):
                for jls1 in range(LS1):
                    for jls01 in range(LS0):
                        for jls11 in range(LS1):
                            # Check that values match
                            if (jls0 // ds0) == (jls1 // ds1) == (jls01 // ds0) == (jls11 // ds1):
                                assert tmp[ib, jls0, jls01, jn, ib, jls1, jls11, jn] == 1, \
                                    "Invalid value at {}".format((ib, jls0, jls01, jn, ib, jls1, jls11, jn))
                            else:
                                assert tmp[ib, jls0, jls01, jn, ib, jls1, jls11, jn] < 1, \
                                    "Invalid value at {}".format((ib, jls0, jls01, jn, ib, jls1, jls11, jn))
    assert torch.allclose(tmp[:, :, :, :, B0:, :, :, :], torch.tensor(0, dtype=torch.int8)), "Invalid values"

    mask = tmp.view(B0, LS0 ** 2, N, B1, LS1 ** 2, N).permute(0, 2, 1, 3, 5, 4)
    # Shape: (B, N, LS0**2, B, N, LS1**2)
    mask = mask.contiguous().view(B0, N * LS0 ** 2, B1, N * LS1 ** 2)

    return mask


def get_standard_instance_mask(batch_size0, batch_size1, pred_step, device="cuda"):
    B0, B1, N = batch_size0, batch_size1, pred_step
    device = torch.device(device)

    assert B0 <= B1, "Invalid B0, B1: {} {}".format(B0, B1)

    # mask meaning: -2: omit, -1: temporal neg (hard), 0: easy neg, 1: pos, -3: spatial neg
    mask = torch.zeros((B0, N, B1, N), dtype=torch.int8, requires_grad=False).detach().to(device)
    # temporal neg pairs
    for k in range(B0):
        mask[k, :, k, :] = -1
    # positive pairs
    for j in range(B0):
        mask[j, torch.arange(N), j, torch.arange(N)] = 1
    for i in range(B0):
        for j in range(N):
            assert mask[i, j, i, j] == 1, "Invalid value at {}, {}".format(i, j)
            for xi in range(B0):
                if i == xi:
                    continue
                for xj in range(N):
                    if j == xj:
                        continue
                    assert mask[i, j, xi, xj] < 1, "Invalid value at {}, {}".format(i, j)
    assert torch.allclose(mask[:, :, B0:, :], torch.tensor(0, dtype=torch.int8)), "Invalid values"

    return mask


def process_mask(mask):
    # dot product is computed in parallel gpus, so get less easy neg, bounded by batch size in each gpu'''
    # mask meaning: -2: omit, -1: temporal neg (hard), 0: easy neg, 1: pos, -3: spatial neg
    target = mask == 1
    # This doesn't seem to cause any issues in our implementation
    target.requires_grad = False
    return target
