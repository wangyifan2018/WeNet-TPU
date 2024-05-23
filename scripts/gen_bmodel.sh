
model_transform.py \
    --model_name encoder \
    --model_def ../onnx/encoder_optimized_float_model.onnx \
    --input_shapes [[1,1,71,80],[1,96,128,128],[1,512,12,6],[1,8,8,136]] \
    --test_input encoder.npz \
    --test_result encoder_top_outputs.npz \
    --mlir encoder.mlir

model_deploy.py \
    --mlir encoder.mlir \
    --quantize F16 \
    --chip bm1684x \
    --model wenet_encoder_f16.bmodel \
    --test_input encoder_in_f32.npz \
    --test_reference encoder_top_outputs.npz \
    --debug

model_transform.py \
    --model_name ctc \
    --model_def ../onnx/ctc_optimized_float_model.onnx \
    --input_shapes [[1,512,1,8]] \
    --test_input ctc.npz \
    --test_result ctc_top_outputs.npz \
    --mlir ctc.mlir

model_deploy.py \
    --mlir ctc.mlir \
    --quantize F16 \
    --chip bm1684x \
    --model wenet_ctc_f16.bmodel \
    --test_input ctc_in_f32.npz \
    --test_reference ctc_top_outputs.npz \
    --compare_all

