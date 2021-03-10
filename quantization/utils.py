from . import scalar


def quantize_ctc_system(system, config):
    print('')
    system.asr_model = quantize_model_scalar(
        system.asr_model,
        params=config.quant_params
    )

    system.task_type_model = quantize_model_scalar(
        system.task_type_model,
        params=config.quant_params
    )


def quantize_model_scalar(model, params):
    print(f'== Quantization for <{model.__class__.__name__}> ==')
    print(f' - Number of bits: {params.bits}')
    print(f' - Noise rate: {params.noise_rate}')

    scalar.quantize_model_(
        model,
        p=params.noise_rate,
        bits=params.bits,
        update_step=1000
    )

    print('==\n')
    return model

    # logger.info(f"QUANTIZATION: Using {num_bits} bits")
    # logger.info(f"QUANTIZATION: Using {quant_noise_scalar} noise rate")
    # logger.info(f"QUANTIZATION: jitter set to {jitter}")
    # if quant_noise_scalar > 0:
    #     # quantize_model edits the model in place
    #     scalar.quantize_model_(model, p=quant_noise_scalar, bits=num_bits, update_step=1000, jitter=jitter)
    return model
