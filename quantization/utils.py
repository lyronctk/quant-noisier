from . import scalar


def quantize_ctc_system(system, config):
    print("== ASR MODEL ==")
    system.asr_model = quantize_model_scalar(
        system.asr_model,
        num_bits=config.quant_params.bits
    )
    print("== ASR MODEL ==")

    print("== TASK TYPE MODEL ==")
    system.task_type_model = quantize_model_scalar(
        system.task_type_model,
        num_bits=config.quant_params.bits
    )
    print("== TASK TYPE MODEL ==")


def quantize_model_scalar(model, num_bits):
    print(f'num_bits: {num_bits}')
    print(model)
    # jitter = getattr(model_cfg, "quant_noise_jitter", False)
    # quant_noise_scalar = getattr(model_cfg, "quant_noise_scalar", 0) or 0
    # logger.info(f"QUANTIZATION: Using {num_bits} bits")
    # logger.info(f"QUANTIZATION: Using {quant_noise_scalar} noise rate")
    # logger.info(f"QUANTIZATION: jitter set to {jitter}")
    # if quant_noise_scalar > 0:
    #     # quantize_model edits the model in place
    #     scalar.quantize_model_(model, p=quant_noise_scalar, bits=num_bits, update_step=1000, jitter=jitter)
    return model
