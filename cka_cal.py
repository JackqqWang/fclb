import torch
import numpy as np
import mmcv
import torch


def similarity_pair_batch_cka(data1, data2, bs=2048):
    feat1 = data1['feat']
    feat2 = data2['feat']
    num_layer1 = len(feat1.keys())
    num_layer2 = len(feat2.keys())
    name1 = data1['model_name']
    name2 = data2['model_name']
    print(f'number of layers in {name1} is {num_layer1}')
    print(f'number of layers in {name2} is {num_layer2}')

    num_sample = list(feat1.values())[0].shape[0]
    num_batch = np.int64(np.ceil(num_sample / bs))
    print(
        f'number of samples {num_sample}, number of batch {num_batch}, batch size {bs}')

    cka_map = torch.zeros((num_batch, num_layer1, num_layer2)).cuda()
    prog_bar = mmcv.ProgressBar(num_batch)
    for b_id in range(num_batch):
        start = b_id*bs
        end = (b_id+1)*bs if (b_id+1)*bs < num_sample-1 else num_sample-1
        for i, (k1, v1) in enumerate(feat1.items()):
            for j, (k2, v2) in enumerate(feat2.items()):
                cka_from_examples = cka_linear_torch(
                    torch.tensor(v1[start:end]).cuda(),
                    torch.tensor(v2[start:end]).cuda())
                cka_map[b_id, i, j] = cka_from_examples
        prog_bar.update()
    return cka_map.mean(0).detach().cpu().numpy()

def cka_linear_torch(x1, x2):
    x1 = gram_linear(rearrange_activations(x1))
    x2 = gram_linear(rearrange_activations(x2))
    similarity = _cka(x1, x2)
    return similarity

def cka_rbf_torch(x1, x2, threshold=1.0):
    x1 = gram_rbf(rearrange_activations(x1), threshold=threshold)
    x2 = gram_rbf(rearrange_activations(x2), threshold=threshold)
    similarity = _cka(x1, x2)
    similarity = round(similarity.item(),4)
    return similarity


def rearrange_activations(activations):
    batch_size = activations.shape[0]
    # flat_activations = activations.view(batch_size, -1) # orginal one
    flat_activations = activations.reshape(batch_size, -1)
    return flat_activations


def gram_linear(x):
    return torch.mm(torch.from_numpy(x), torch.from_numpy(x.T))

def gram_rbf(x, threshold=1.0):
    """Compute Gram (kernel) matrix for an RBF kernel.

    Args:
        x: A num_examples x num_features matrix of features.
        threshold: Fraction of median Euclidean distance to use as RBF kernel
        bandwidth. (This is the heuristic we use in the paper. There are other
        possible ways to set the bandwidth; we didn't try them.)

    Returns:
        A num_examples x num_examples Gram matrix of examples.
    """
    dot_products = x.mm(x.T)
    sq_norms = torch.diag(dot_products)
    sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
    sq_median_distance = torch.median(sq_distances)
    return torch.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))


def center_gram(gram, unbiased=False):
    if not torch.allclose(gram, gram.T):
        raise ValueError('Input must be a symmetric matrix.')

    if unbiased:
        pass
        # TODO
    else:
        means = torch.mean(gram, dim=0, dtype=torch.float64) # , dtype=torch.float64
        means -= torch.mean(means) / 2
        gram -= torch.unsqueeze(means, len(means.shape))
        gram -= torch.unsqueeze(means, 0)

    return gram


def _cka(gram_x, gram_y, debiased=False):
    gram_x = center_gram(gram_x, unbiased=debiased)
    gram_y = center_gram(gram_y, unbiased=debiased)

    scaled_hsic = (gram_x.view(-1) * gram_y.view(-1)).sum()
    # normalization_x = torch.linalg.norm(gram_x)
    # normalization_y = torch.linalg.norm(gram_y)
    normalization_x = torch.norm(gram_x)
    normalization_y = torch.norm(gram_y)

    return scaled_hsic / (normalization_x * normalization_y)