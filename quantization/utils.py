import logging
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
    logging.info(f'== Model <{model.__class__.__name__}> ==')

    scalar.quantize_model_(
        model,
        p=params.noise_rate,
        bits=params.bits,
        update_step=params.update_step,
        method=params.method,
        jitter=params.jitter,
        qn_lambda=params.qn_lambda
    )

    logging.info('==\n' )
    return model
