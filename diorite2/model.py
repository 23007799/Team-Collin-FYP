# model.py
from tsfm_public import TinyTimeMixerForPrediction, count_parameters

def load_model(tsp, model_path, revision, context, forecast):
    prediction_filter_length = min(8, forecast - 1)

    model = TinyTimeMixerForPrediction.from_pretrained(
        model_path,
        revision=revision,
        context_length=context,
        prediction_filter_length=prediction_filter_length,
        num_input_channels=tsp.num_input_channels,
        decoder_mode="mix_channel",
        prediction_channel_indices=tsp.prediction_channel_indices,
        exogenous_channel_indices=tsp.exogenous_channel_indices,
        categorical_vocab_size_list=tsp.categorical_vocab_size_list,
        enable_forecast_channel_mixing=True,
    )

    print("Params:", count_parameters(model))
    return model
