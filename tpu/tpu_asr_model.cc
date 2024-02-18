#include "tpu/tpu_asr_model.h"
#include <algorithm>
#include <memory>
#include <utility>
#include "utils/string.h"

namespace wenet {

  void TPUAsrModel::GetInputOutputInfo(
    const std::vector<bmruntime::Tensor*> input,
    const std::vector<bmruntime::Tensor*> output) {
    // Input info
    for (size_t i = 0; i < input.size(); ++i) {
      auto& shapes = input[i]->tensor()->shape;
      LOG(INFO) << "\tInput-" << i << ": Shape [" << shapes.dims[0] << ","
        << shapes.dims[1] << "," << shapes.dims[2] << ","
        << shapes.dims[3] << "]";
    }
    // Output info
    for (size_t i = 0; i < output.size(); ++i) {
      auto& shapes = output[i]->tensor()->shape;
      LOG(INFO) << "\tOutput-" << i << ": Shape [" << shapes.dims[0] << ","
        << shapes.dims[1] << "," << shapes.dims[2] << ","
        << shapes.dims[3] << "]";
    }
  }

  void TPUAsrModel::Read(const std::string& model_dir) {
    std::string encoder_model_path = model_dir + "/encoder.bmodel";
    std::string ctc_model_path = model_dir + "/ctc.bmodel";

    // 0. Init context, dev_id default 0
    int dev_id = 0;
    encoder_ctx_ = std::make_shared<bmruntime::Context>(dev_id);
    bm_status_t status = encoder_ctx_->load_bmodel(encoder_model_path.c_str());
    assert(BM_SUCCESS == status);

    ctc_ctx_ = std::make_shared<bmruntime::Context>(dev_id);
    status = ctc_ctx_->load_bmodel(ctc_model_path.c_str());
    assert(BM_SUCCESS == status);

    // 1. Load models
    std::vector<const char*> network_names;
    encoder_ctx_->get_network_names(&network_names);
    encoder_net_ = std::make_shared<bmruntime::Network>(
      *encoder_ctx_, network_names[0], 0);  // use stage[0]
    assert(encoder_net_->info()->input_num == 4);

    ctc_ctx_->get_network_names(&network_names);
    ctc_net_ = std::make_shared<bmruntime::Network>(*ctc_ctx_, network_names[0],
      0);  // use stage[0]
    assert(ctc_net_->info()->input_num == 1);

    // 2. Init input/output tensors
    AllocMemory(encoder_net_, &encoder_input_, &encoder_output_);
    AllocMemory(ctc_net_, &ctc_input_, &ctc_output_);
    Reset();

    // 3. Read model input/output nodes
    LOG(INFO) << "TPU Encoder:";
    GetInputOutputInfo(encoder_input_, encoder_output_);
    LOG(INFO) << "TPU CTC:";
    GetInputOutputInfo(ctc_input_, ctc_output_);

    // 4. Parse metadatas
    right_context_ = 14;    //  Only support 1/8 subsample, since
    subsampling_rate_ = 8;  //   1/4 subsample is too slow on edge-devices.
    sos_ = ctc_output_[0]->tensor()->shape.dims[1] - 1;
    eos_ = sos_;
    chunk_size_ = ctc_input_[0]->tensor()->shape.dims[3];
    num_left_chunks_ =
      encoder_input_[3]->tensor()->shape.dims[3] / chunk_size_ - 1;
    hidden_dim_ = ctc_input_[0]->tensor()->shape.dims[1];
    int frames =
      (chunk_size_ - 1) * subsampling_rate_ + right_context_ + 1;  // 71
    CHECK_EQ(frames, encoder_input_[0]->tensor()->shape.dims[2])
      << " Only support 1/8 subsample, since 1/4 subsample"
      << " is too slow on edge-devices.";
    LOG(INFO) << "TPU Model Info:";
    LOG(INFO) << "\tchunk_size " << chunk_size_;
    LOG(INFO) << "\tnum_left_chunks " << num_left_chunks_;
    LOG(INFO) << "\tsubsampling_rate " << subsampling_rate_;
    LOG(INFO) << "\tright context " << right_context_;
    LOG(INFO) << "\tsos " << sos_;
    LOG(INFO) << "\teos " << eos_;
    LOG(INFO) << "\tis bidirectional decoder " << is_bidirectional_decoder_;
    LOG(INFO) << "\thidden_dim " << hidden_dim_;
  }

  TPUAsrModel::TPUAsrModel(const TPUAsrModel& other) {
    // metadatas (BaseClass)
    right_context_ = other.right_context_;
    subsampling_rate_ = other.subsampling_rate_;
    sos_ = other.sos_;
    eos_ = other.eos_;
    is_bidirectional_decoder_ = other.is_bidirectional_decoder_;
    chunk_size_ = other.chunk_size_;
    num_left_chunks_ = other.num_left_chunks_;
    offset_ = other.offset_;
    // metadatas (ChileClass)
    hidden_dim_ = other.hidden_dim_;
    chunk_id_ = other.chunk_id_;
    // models, in/out tensors are not copied here.
    encoder_ctx_ = std::move(other.encoder_ctx_);
    ctc_ctx_ = std::move(other.ctc_ctx_);
    encoder_net_ = std::move(other.encoder_net_);
    ctc_net_ = std::move(other.ctc_net_);
  }

  std::shared_ptr<AsrModel> TPUAsrModel::Copy() const {
    auto asr_model = std::make_shared<TPUAsrModel>(*this);
    // Reset the inner states for new decoding
    asr_model->AllocMemory(encoder_net_, &(asr_model->encoder_input_),
      &(asr_model->encoder_output_));
    asr_model->AllocMemory(ctc_net_, &(asr_model->ctc_input_),
      &(asr_model->ctc_output_));

    asr_model->Reset();
    return std::move(asr_model);
  }

  void TPUAsrModel::AllocMemory(const std::shared_ptr<bmruntime::Network>& model,
    std::vector<bmruntime::Tensor*>* inputs,
    std::vector<bmruntime::Tensor*>* outputs) {
    *inputs = std::move(model->Inputs());
    *outputs = std::move(model->Outputs());
  }

  void TPUAsrModel::Reset() {
    offset_ = 0;
    chunk_id_ = 0;
    cached_feature_.clear();
    encoder_outs_.clear();
    encoder_outs_.resize(hidden_dim_);  // [512][0~MaxFrames]
    ResetDevMem();
  }

  void TPUAsrModel::ForwardEncoderFunc(
    const std::vector<std::vector<float>>& chunk_feats,
    std::vector<std::vector<float>>* out_prob) {
    // 1. Forward Encoder
    PrepareEncoderInput(chunk_feats);

    assert(encoder_net_->info()->input_num == 4);
    LOG(INFO) << "\twenet encoder inference start ";
    auto status = encoder_net_->Forward();
    assert(BM_SUCCESS == status);

    // 2. Forward CTC
    PrepareCtcInput();
    LOG(INFO) << "\twenet ctc inference ";
    status = ctc_net_->Forward();
    assert(BM_SUCCESS == status);

    // 3. Extract final outout_prob
    void* probs_Concat_f32 =
      calloc(ctc_output_[0]->num_elements(), sizeof(float));

    ctc_output_[0]->CopyTo(probs_Concat_f32);
    const float* raw_data = static_cast<float*>(probs_Concat_f32);

    out_prob->resize(chunk_size_);  // v[16][4233]
    for (auto& val : *out_prob) {
      val.clear();
      val.reserve(eos_ + 1);
    }
    for (size_t idx = 0, i = 0; i < static_cast<size_t>(eos_ + 1); ++i) {
      for (size_t j = 0; j < static_cast<size_t>(chunk_size_); ++j) {
        (*out_prob)[j].emplace_back(raw_data[idx++]);
      }
    }

    free(probs_Concat_f32);
    // TODO: 4. Forward Decoder.
    //  update encoder_outs_ here.
  }

  void TPUAsrModel::PrepareEncoderInput(
    const std::vector<std::vector<float>>& chunk_feats) {
    chunk_id_ += 1;

    // 1. input-0: chunk
    float* feat_ptr =
      (float*)calloc(encoder_input_[0]->num_elements(), sizeof(float));
    int64_t addr_shift = 0;
    for (size_t i = 0; i < cached_feature_.size(); ++i) {  // copy cached_feature_
      memcpy(feat_ptr + addr_shift, cached_feature_[i].data(),
        cached_feature_[i].size() * sizeof(float));
      addr_shift += cached_feature_[i].size();
    }

    for (size_t i = 0; i < chunk_feats.size(); ++i) {  // copy chunk_feats
      memcpy(feat_ptr + addr_shift, chunk_feats[i].data(),
        chunk_feats[i].size() * sizeof(float));
      addr_shift += chunk_feats[i].size();
    }
    encoder_input_[0]->CopyFrom((void*)feat_ptr);

    // 2. att_cache & cnn_cache
    void* r_att_cache_Concat_f32 =
      calloc(encoder_output_[1]->num_elements(), sizeof(float));
    assert(BM_SUCCESS == MemCopyD2D(encoder_ctx_->handle(), encoder_input_[1], encoder_output_[1], encoder_output_[1]->ByteSize()));

    void* r_cnn_cache_Conv_f32 =
      calloc(encoder_output_[2]->num_elements(), sizeof(float));
    assert(BM_SUCCESS == MemCopyD2D(encoder_ctx_->handle(), encoder_input_[2], encoder_output_[2], encoder_output_[2]->ByteSize()));

    // 3. att_mask
    // For last chunk_feats whose size < chunk_size * subsampling,
    //  we will do nothing since it hardly affects wer even if we
    //  use `wrong` att_mask where trailing zeros are not masked.
    void* att_mask = calloc(encoder_input_[3]->num_elements(), sizeof(float));
    encoder_input_[3]->CopyTo(att_mask);
    int valid_len = chunk_id_ * chunk_size_;
    int total_len = (num_left_chunks_ + 1) * chunk_size_;
    int head = encoder_input_[3]->tensor()->shape.dims[1];
    if (valid_len <= total_len) {
      std::vector<float> padding(total_len, 1.0f);
      for (size_t i = 0; i < static_cast<size_t>(total_len - valid_len); ++i) {
        padding[i] = 0.0f;
      }
      for (size_t i = 0; i < static_cast<size_t>(head * chunk_size_); ++i) {
        float* start_ptr = static_cast<float*>(att_mask) + total_len * i;
        memcpy(start_ptr, padding.data(), total_len * sizeof(float));
      }
    }
    encoder_input_[3]->CopyFrom(att_mask);

    // release
    free(feat_ptr);
    free(r_att_cache_Concat_f32);
    free(r_cnn_cache_Conv_f32);
    free(att_mask);
  }

  void TPUAsrModel::PrepareCtcInput() {
    // 1. chunk_out
    void* output_Unsqueeze_f32 =
      calloc(encoder_output_[0]->num_elements(), sizeof(float));
    assert(BM_SUCCESS == MemCopyD2D(ctc_ctx_->handle(), ctc_input_[0], encoder_output_[0], encoder_output_[0]->ByteSize()));
    free(output_Unsqueeze_f32);
  }

  float TPUAsrModel::ComputeAttentionScore(const float* prob,
    const std::vector<int>& hyp, int eos,
    int decode_out_len) {
    // TODO: Support decoder.
    //  Currently, running decoder on edge-devices is time-consuming since the
    //  the length of input is much longer than encoder. To achieve a better
    //  accuracy-speed trade-off, we disable rescoring by default.
    return 0.0;
  }

  void TPUAsrModel::AttentionRescoring(const std::vector<std::vector<int>>& hyps,
    float reverse_weight,
    std::vector<float>* rescoring_score) {
    // TODO: Support decoder.
    //  Currently, running decoder on edge-devices is time-consuming since the
    //  the length of input is much longer than encoder. To achieve a better
    //  accuracy-speed trade-off, we disable rescoring by default.
    LOG(INFO) << "Skip rescore. Please set rescoring_weight = 0.0";
  }

  bm_status_t TPUAsrModel::MemCopyD2D(bm_handle_t handle, const bmruntime::Tensor* dst, const bmruntime::Tensor* src, size_t size) {
    return bm_memcpy_d2d_byte(handle, dst->tensor()->device_mem, 0, src->tensor()->device_mem, 0, size);
  }

  void TPUAsrModel::ResetDevMem() {
    void* chunk = calloc(encoder_input_[0]->num_elements(), sizeof(float));
    void* att_cache = calloc(encoder_input_[1]->num_elements(), sizeof(float));
    void* cnn_cache = calloc(encoder_input_[2]->num_elements(), sizeof(float));
    void* att_mask = calloc(encoder_input_[3]->num_elements(), sizeof(float));
    void* output_Reshape = calloc(encoder_output_[0]->num_elements(), sizeof(float));
    void* r_att_cache_Concat = calloc(encoder_output_[1]->num_elements(), sizeof(float));
    void* r_cnn_cache_Conv = calloc(encoder_output_[2]->num_elements(), sizeof(float));
    encoder_input_[0]->CopyFrom(chunk);
    encoder_input_[1]->CopyFrom(att_cache);
    encoder_input_[2]->CopyFrom(cnn_cache);
    encoder_input_[3]->CopyFrom(att_mask);
    encoder_output_[0]->CopyFrom(output_Reshape);
    encoder_output_[1]->CopyFrom(r_att_cache_Concat);
    encoder_output_[2]->CopyFrom(r_cnn_cache_Conv);
    free(chunk);
    free(att_cache);
    free(cnn_cache);
    free(att_mask);
    free(output_Reshape);
    free(r_att_cache_Concat);
    free(r_cnn_cache_Conv);
  }

  TPUAsrModel::~TPUAsrModel() {
    ResetDevMem();
  }
}  // namespace wenet
