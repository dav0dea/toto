import torch

DEVICE = "cuda"
toto = None


def load_model():
    from toto.model.toto import Toto

    global toto
    toto = Toto.from_pretrained("Datadog/Toto-Open-Base-1.0").to(DEVICE)
    toto.eval()
    toto.use_memory_efficient = True
    toto.compile()


@torch.inference_mode()
def embed(inputs: torch.Tensor, global_average: bool = False) -> torch.Tensor:
    """
    Embed the input time series using the Toto model (https://huggingface.co/Datadog/Toto-Open-Base-1.0).
    Args:
        inputs (torch.Tensor): Input time series data, shape `(num_channels, sequence_length)` or `(batch_size, num_channels, sequence_length)`.
        global_average (bool): If True, applies global average pooling to the output embeddings. Defaults to False.
    Returns:
        torch.Tensor: The embedded time series data, shape `(batch_size, num_channels, num_patches, embedding_dim)` or
        `(num_channels, num_patches, embedding_dim)` if no batch dimension is present. If `global_average` is True,
        the output will be of shape `(batch_size, embedding_dim)` or `(embedding_dim,)`.
    """
    global toto
    if toto is None:
        load_model()

    if not isinstance(inputs, torch.Tensor):
        inputs = torch.tensor(inputs, dtype=torch.float32).to(DEVICE)

    has_batch_dim = True
    if inputs.ndim < 3:
        has_batch_dim = False
        inputs = inputs.unsqueeze(0)

    input_padding_mask, id_mask = torch.full_like(inputs, True, dtype=torch.bool), torch.zeros_like(inputs)

    # Standard scaling operation, same API but without ID mask.
    scaled_inputs, loc, scale = toto.model.scaler(
        inputs,
        weights=torch.ones_like(inputs, device=inputs.device),
        padding_mask=input_padding_mask,
        prefix_length=None,
    )

    embeddings, reduced_id_mask = toto.model.patch_embed(scaled_inputs, id_mask)

    # Apply the transformer on the embeddings
    transformed = toto.model.transformer(embeddings, reduced_id_mask, None)

    if global_average:
        transformed = transformed.mean(dim=(1, 2))

    if has_batch_dim:
        return transformed.cpu()
    else:
        return transformed.squeeze(0).cpu()
