#ifndef WENET_TPU_TPU_TPU_ASR_MODEL_H_
#define WENET_TPU_TPU_TPU_ASR_MODEL_H_

#include <memory>
#include <string>
#include <vector>
#include "bmruntime_cpp.h"
#include "decoder/asr_model.h"
#include "utils/log.h"
#include "utils/utils.h"
namespace wenet {

  class TPUAsrModel : public AsrModel {
  public:
    TPUAsrModel() = default;
    ~TPUAsrModel();
    TPUAsrModel(const TPUAsrModel& other);
    void Read(const std::string& model_dir);
    void Reset() override;
    void AttentionRescoring(const std::vector<std::vector<int>>& hyps,
      float reverse_weight,
      std::vector<float>* rescoring_score) override;
    std::shared_ptr<AsrModel> Copy() const override;
    static void AllocMemory(const std::shared_ptr<bmruntime::Network>& model,
      std::vector<bmruntime::Tensor*>* input,
      std::vector<bmruntime::Tensor*>* output);
    void GetInputOutputInfo(const std::vector<bmruntime::Tensor*> input_tensors,
      const std::vector<bmruntime::Tensor*> output_tensors);
    void PrepareEncoderInput(const std::vector<std::vector<float>>& chunk_feats);
    void PrepareCtcInput();

  protected:
    void ForwardEncoderFunc(const std::vector<std::vector<float>>& chunk_feats,
      std::vector<std::vector<float>>* ctc_prob) override;

    float ComputeAttentionScore(const float* prob, const std::vector<int>& hyp,
      int eos, int decode_out_len);

    bm_status_t MemCopyD2D(bm_handle_t handle, const bmruntime::Tensor* dst, const bmruntime::Tensor* src, size_t size);

    void ResetDevMem();

  private:
    // metadatas
    int hidden_dim_ = 512;
    int chunk_id_ = 0;

    // models
    std::shared_ptr<bmruntime::Context> encoder_ctx_ = nullptr;
    std::shared_ptr<bmruntime::Context> ctc_ctx_ = nullptr;
    std::shared_ptr<bmruntime::Network> encoder_net_ = nullptr;
    std::shared_ptr<bmruntime::Network> ctc_net_ = nullptr;

    // input/output tensors
    std::vector<bmruntime::Tensor*> encoder_input_, encoder_output_;
    std::vector<bmruntime::Tensor*> ctc_input_, ctc_output_;
    std::vector<std::vector<float>> encoder_outs_;
  };

}  // namespace wenet

#endif  // WENET_TPU_TPU_TPU_ASR_MODEL_H_
