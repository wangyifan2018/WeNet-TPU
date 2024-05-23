# WeNet & Sophgo TPU (BM1684X)

### Setup
* Step 1. Setup environment. It depends on Sophgo driver.
```
# libsophon doc
https://doc.sophgo.com/sdk-docs/v23.09.01-lts/docs_latest_release/docs/libsophon/guide/html/1_install.html
```

* Step 2. Build decoder_main. It requires cmake 3.14 or above. (~3min)

``` sh
# Assume current dir is `Wenet-TPU`
# Local compile, for PCIE mode
cmake -B build -DTPU=ON -DONNX=OFF -DTORCH=OFF -DWEBSOCKET=OFF -DGRPC=OFF
cmake --build build -j$(nproc)

# Cross compile, for SOC mode
# install if need
sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
# set SDK path
cmake -B build -DTPU=ON -DONNX=OFF -DTORCH=OFF -DWEBSOCKET=OFF -DGRPC=OFF -DSDK=/path/to/soc-sdk -DCMAKE_TOOLCHAIN_FILE=toolchains/aarch64-linux-gnu.toolchain.cmake
cmake --build build -j$(nproc)
# copy build/bin/decoder_main to device
```

* Step 3. Download the model and place it in the `bmodel` folder.

Download from sftp server
``` sh
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
python3 -m dfss --url=open@sophgo.com:ezoo/wenet/bmodel.zip
unzip bmodel.zip

./bmodel
├── ctc.bmodel
├── encoder.bmodel
└── units.txt
```

[optional] compile by yourself, setup [tpu-mlr](https://github.com/sophgo/tpu-mlir) and run ./scripts/gen_bmodel.sh

``` sh
python3 -m dfss --url=open@sophgo.com:ezoo/wenet/onnx.zip
unzip onnx.zip

./onnx
├── ctc_optimized_float_model.onnx
└── encoder_optimized_float_model.onnx

cd scripts
./gen_bmodel.sh
```

* Step 4. Testing on Sophgo BM1684X(PCIE/SOC), the RTF(real time factor) is shown in console.

``` sh
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
export GLOG_logtostderr=1
export GLOG_v=3
./build/bin/decoder_main \
    --chunk_size 8 \
    --num_left_chunks 16 \
    --rescoring_weight 0.0 \
    --wav_path ./data/test.wav \
    --tpu_model_dir ./bmodel \
    --unit_path ./bmodel/units.txt \
    --continuous_decoding 2>&1 | tee log.txt

I0219 14:31:10.723145 2508725 params.h:224] Reading SOPHGO TPU model from ./bmodel
I0219 14:31:11.230612 2508725 tpu_asr_model.cc:60] TPU Encoder:
I0219 14:31:11.230625 2508725 tpu_asr_model.cc:15]      Input-0: Shape [1,1,71,80]
I0219 14:31:11.230626 2508725 tpu_asr_model.cc:15]      Input-1: Shape [1,96,128,128]
I0219 14:31:11.230628 2508725 tpu_asr_model.cc:15]      Input-2: Shape [1,512,12,6]
I0219 14:31:11.230628 2508725 tpu_asr_model.cc:15]      Input-3: Shape [1,8,8,136]
I0219 14:31:11.230628 2508725 tpu_asr_model.cc:22]      Output-0: Shape [1,512,1,8]
I0219 14:31:11.230629 2508725 tpu_asr_model.cc:22]      Output-1: Shape [1,96,128,128]
I0219 14:31:11.230630 2508725 tpu_asr_model.cc:22]      Output-2: Shape [1,512,12,6]
I0219 14:31:11.230631 2508725 tpu_asr_model.cc:62] TPU CTC:
I0219 14:31:11.230633 2508725 tpu_asr_model.cc:15]      Input-0: Shape [1,512,1,8]
I0219 14:31:11.230633 2508725 tpu_asr_model.cc:22]      Output-0: Shape [1,5538,1,8]
I0219 14:31:11.230634 2508725 tpu_asr_model.cc:79] TPU Model Info:
I0219 14:31:11.230634 2508725 tpu_asr_model.cc:80]      chunk_size 8
I0219 14:31:11.230635 2508725 tpu_asr_model.cc:81]      num_left_chunks 16
I0219 14:31:11.230636 2508725 tpu_asr_model.cc:82]      subsampling_rate 8
I0219 14:31:11.230636 2508725 tpu_asr_model.cc:83]      right context 14
I0219 14:31:11.230638 2508725 tpu_asr_model.cc:84]      sos 5537
I0219 14:31:11.230638 2508725 tpu_asr_model.cc:85]      eos 5537
I0219 14:31:11.230638 2508725 tpu_asr_model.cc:86]      is bidirectional decoder 0
I0219 14:31:11.230639 2508725 tpu_asr_model.cc:87]      hidden_dim 512
I0219 14:31:11.230641 2508725 params.h:236] Reading unit table ./bmodel/units.txt
I0219 14:31:11.242403 2508727 decoder_main.cc:54] num frames 418
I0219 14:31:11.260071 2508727 asr_decoder.cc:104] Required 71 get 71
I0219 14:31:11.260669 2508727 tpu_asr_model.cc:145]     wenet encoder inference start
I0219 14:31:11.274698 2508727 tpu_asr_model.cc:151]     wenet ctc inference
I0219 14:31:11.276538 2508727 asr_decoder.cc:119] forward takes 15 ms, search takes 1 ms
I0219 14:31:11.276819 2508727 asr_decoder.cc:104] Required 64 get 64
I0219 14:31:11.277395 2508727 tpu_asr_model.cc:145]     wenet encoder inference start
I0219 14:31:11.291445 2508727 tpu_asr_model.cc:151]     wenet ctc inference
I0219 14:31:11.293371 2508727 asr_decoder.cc:119] forward takes 15 ms, search takes 1 ms
I0219 14:31:11.293514 2508727 asr_decoder.cc:200] Partial CTC result 甚至出
I0219 14:31:11.293520 2508727 decoder_main.cc:72] Partial result: 甚至出
I0219 14:31:11.293530 2508727 asr_decoder.cc:104] Required 64 get 64
I0219 14:31:11.294103 2508727 tpu_asr_model.cc:145]     wenet encoder inference start
I0219 14:31:11.308060 2508727 tpu_asr_model.cc:151]     wenet ctc inference
I0219 14:31:11.309971 2508727 asr_decoder.cc:119] forward takes 15 ms, search takes 1 ms
I0219 14:31:11.310119 2508727 asr_decoder.cc:200] Partial CTC result 甚至出现交
I0219 14:31:11.310125 2508727 decoder_main.cc:72] Partial result: 甚至出现交
I0219 14:31:11.310135 2508727 asr_decoder.cc:104] Required 64 get 64
I0219 14:31:11.310704 2508727 tpu_asr_model.cc:145]     wenet encoder inference start
I0219 14:31:11.324647 2508727 tpu_asr_model.cc:151]     wenet ctc inference
I0219 14:31:11.326591 2508727 asr_decoder.cc:119] forward takes 15 ms, search takes 1 ms
I0219 14:31:11.326766 2508727 asr_decoder.cc:200] Partial CTC result 甚至出现交易几乎
I0219 14:31:11.326774 2508727 decoder_main.cc:72] Partial result: 甚至出现交易几乎
I0219 14:31:11.326786 2508727 asr_decoder.cc:104] Required 64 get 64
I0219 14:31:11.327360 2508727 tpu_asr_model.cc:145]     wenet encoder inference start
I0219 14:31:11.341286 2508727 tpu_asr_model.cc:151]     wenet ctc inference
I0219 14:31:11.343219 2508727 asr_decoder.cc:119] forward takes 15 ms, search takes 1 ms
I0219 14:31:11.343372 2508727 asr_decoder.cc:200] Partial CTC result 甚至出现交易几乎停滞的
I0219 14:31:11.343377 2508727 decoder_main.cc:72] Partial result: 甚至出现交易几乎停滞的
I0219 14:31:11.343387 2508727 asr_decoder.cc:104] Required 64 get 64
I0219 14:31:11.343955 2508727 tpu_asr_model.cc:145]     wenet encoder inference start
I0219 14:31:11.357940 2508727 tpu_asr_model.cc:151]     wenet ctc inference
I0219 14:31:11.359941 2508727 asr_decoder.cc:119] forward takes 15 ms, search takes 1 ms
I0219 14:31:11.360111 2508727 asr_decoder.cc:200] Partial CTC result 甚至出现交易几乎停滞的情况
I0219 14:31:11.360116 2508727 decoder_main.cc:72] Partial result: 甚至出现交易几乎停滞的情况
I0219 14:31:11.360126 2508727 asr_decoder.cc:104] Required 64 get 27
I0219 14:31:11.360697 2508727 tpu_asr_model.cc:145]     wenet encoder inference start
I0219 14:31:11.374732 2508727 tpu_asr_model.cc:151]     wenet ctc inference
I0219 14:31:11.376614 2508727 asr_decoder.cc:119] forward takes 15 ms, search takes 1 ms
I0219 14:31:11.376785 2508727 asr_decoder.cc:200] Partial CTC result 甚至出现交易几乎停滞的情况
I0219 14:31:11.376953 2508727 asr_decoder.cc:200] Partial CTC result 甚至出现交易几乎停滞的情况
I0219 14:31:11.376955 2508727 asr_decoder.cc:84] Rescoring cost latency: 0ms.
I0219 14:31:11.376956 2508727 decoder_main.cc:72] Partial result: 甚至出现交易几乎停滞的情况
I0219 14:31:11.376957 2508727 decoder_main.cc:104] test Final result: 甚至出现交易几乎停滞的情况
I0219 14:31:11.376960 2508727 decoder_main.cc:105] Decoded 4203ms audio taken 112ms.
[BMRT][bmcpu_setup:435] INFO:cpu_lib 'libcpuop.so' is loaded.
bmcpu init: skip cpu_user_defined
open usercpu.so, init user_cpu_init
[BMRT][BMProfile:59] INFO:Profile For arch=3
[BMRT][BMProfileDeviceBase:190] INFO:gdma=0, tiu=0, mcu=0
[BMRT][load_bmodel:1594] INFO:Loading bmodel from [./bmodel/encoder.bmodel]. Thanks for your patience...
[BMRT][load_bmodel:1503] INFO:pre net num: 0, load net num: 1
[BMRT][load_tpu_module:1575] INFO:loading firmare in bmodel
[BMRT][bmcpu_setup:435] INFO:cpu_lib 'libcpuop.so' is loaded.
bmcpu init: skip cpu_user_defined
open usercpu.so, init user_cpu_init
[BMRT][BMProfile:59] INFO:Profile For arch=3
[BMRT][BMProfileDeviceBase:190] INFO:gdma=0, tiu=0, mcu=0
[BMRT][load_bmodel:1594] INFO:Loading bmodel from [./bmodel/ctc.bmodel]. Thanks for your patience...
[BMRT][load_bmodel:1503] INFO:pre net num: 0, load net num: 1
[BMRT][load_tpu_module:1575] INFO:loading firmare in bmodel
test 甚至出现交易几乎停滞的情况
I0219 14:31:11.397114 2508725 decoder_main.cc:180] Total: decoded 4203ms audio taken 112ms.
I0219 14:31:11.397120 2508725 decoder_main.cc:182] RTF: 0.02665
```

### Server and client
* Step 1. Start websocket server
```sh
# set -DWEBSOCKET=ON
cmake -B build -DTPU=ON -DONNX=OFF -DTORCH=OFF -DWEBSOCKET=ON -DGRPC=OFF
cmake --build build -j$(nproc)
export GLOG_logtostderr=1
export GLOG_v=3
./build/bin/websocket_server_main \
    --chunk_size 8 \
    --num_left_chunks 16 \
    --rescoring_weight 0.0 \
    --tpu_model_dir ./bmodel \
    --unit_path ./bmodel/units.txt 2>&1 | tee log.txt
I0218 14:05:29.312755 2256320 params.h:224] Reading SOPHGO TPU model from ./bmodel
[BMRT][bmcpu_setup:435] INFO:cpu_lib 'libcpuop.so' is loaded.
bmcpu init: skip cpu_user_defined
open usercpu.so, init user_cpu_init
[BMRT][BMProfile:59] INFO:Profile For arch=3
[BMRT][BMProfileDeviceBase:190] INFO:gdma=0, tiu=0, mcu=0
[BMRT][load_bmodel:1594] INFO:Loading bmodel from [./bmodel/encoder.bmodel]. Thanks for your patience...
[BMRT][load_bmodel:1503] INFO:pre net num: 0, load net num: 1
[BMRT][load_tpu_module:1575] INFO:loading firmare in bmodel
[BMRT][bmcpu_setup:435] INFO:cpu_lib 'libcpuop.so' is loaded.
bmcpu init: skip cpu_user_defined
open usercpu.so, init user_cpu_init
[BMRT][BMProfile:59] INFO:Profile For arch=3
[BMRT][BMProfileDeviceBase:190] INFO:gdma=0, tiu=0, mcu=0
[BMRT][load_bmodel:1594] INFO:Loading bmodel from [./bmodel/ctc.bmodel]. Thanks for your patience...
[BMRT][load_bmodel:1503] INFO:pre net num: 0, load net num: 1
[BMRT][load_tpu_module:1575] INFO:loading firmare in bmodel
I0218 14:05:29.980463 2256320 tpu_asr_model.cc:60] TPU Encoder:
I0218 14:05:29.980515 2256320 tpu_asr_model.cc:15]      Input-0: Shape [1,1,71,80]
I0218 14:05:29.980526 2256320 tpu_asr_model.cc:15]      Input-1: Shape [1,96,128,128]
I0218 14:05:29.980535 2256320 tpu_asr_model.cc:15]      Input-2: Shape [1,512,12,6]
I0218 14:05:29.980542 2256320 tpu_asr_model.cc:15]      Input-3: Shape [1,8,8,136]
I0218 14:05:29.980551 2256320 tpu_asr_model.cc:22]      Output-0: Shape [1,512,1,8]
I0218 14:05:29.980558 2256320 tpu_asr_model.cc:22]      Output-1: Shape [1,96,128,128]
I0218 14:05:29.980566 2256320 tpu_asr_model.cc:22]      Output-2: Shape [1,512,12,6]
I0218 14:05:29.980574 2256320 tpu_asr_model.cc:62] TPU CTC:
I0218 14:05:29.980584 2256320 tpu_asr_model.cc:15]      Input-0: Shape [1,512,1,8]
I0218 14:05:29.980594 2256320 tpu_asr_model.cc:22]      Output-0: Shape [1,5538,1,8]
I0218 14:05:29.980607 2256320 tpu_asr_model.cc:79] TPU Model Info:
I0218 14:05:29.980615 2256320 tpu_asr_model.cc:80]      chunk_size 8
I0218 14:05:29.980626 2256320 tpu_asr_model.cc:81]      num_left_chunks 16
I0218 14:05:29.980634 2256320 tpu_asr_model.cc:82]      subsampling_rate 8
I0218 14:05:29.980643 2256320 tpu_asr_model.cc:83]      right context 14
I0218 14:05:29.980648 2256320 tpu_asr_model.cc:84]      sos 5537
I0218 14:05:29.980655 2256320 tpu_asr_model.cc:85]      eos 5537
I0218 14:05:29.980662 2256320 tpu_asr_model.cc:86]      is bidirectional decoder 0
I0218 14:05:29.980669 2256320 tpu_asr_model.cc:87]      hidden_dim 512
I0218 14:05:29.980677 2256320 params.h:236] Reading unit table ./bmodel/units.txt
I0218 14:05:29.989405 2256320 websocket_server_main.cc:31] Listening at port 10086
I0218 14:05:32.534153 2256347 websocket_server.cc:212] {"signal":"start","nbest":1,"continuous_decoding":true}
I0218 14:05:32.534245 2256347 websocket_server.cc:43] Received speech start signal, start reading speech
I0218 14:05:32.577821 2256347 websocket_server.cc:90] Received 8000 samples
I0218 14:05:33.037973 2256347 websocket_server.cc:90] Received 8000 samples
I0218 14:05:33.043511 2256349 asr_decoder.cc:104] Required 71 get 71
I0218 14:05:33.045230 2256349 tpu_asr_model.cc:145]     wenet encoder inference start
I0218 14:05:33.059576 2256349 tpu_asr_model.cc:151]     wenet ctc inference
I0218 14:05:33.066959 2256349 asr_decoder.cc:119] forward takes 17 ms, search takes 5 ms
I0218 14:05:33.539042 2256347 websocket_server.cc:90] Received 8000 samples
I0218 14:05:33.544203 2256349 asr_decoder.cc:104] Required 64 get 64
I0218 14:05:33.545972 2256349 tpu_asr_model.cc:145]     wenet encoder inference start
I0218 14:05:33.560317 2256349 tpu_asr_model.cc:151]     wenet ctc inference
I0218 14:05:33.565702 2256349 asr_decoder.cc:119] forward takes 17 ms, search takes 4 ms
I0218 14:05:33.566334 2256349 asr_decoder.cc:200] Partial CTC result 甚至出
I0218 14:05:33.566395 2256349 websocket_server.cc:65] Partial result: [{"sentence":"甚至出"}]
I0218 14:05:34.081866 2256347 websocket_server.cc:90] Received 8000 samples
I0218 14:05:34.541147 2256347 websocket_server.cc:90] Received 8000 samples
I0218 14:05:34.546483 2256349 asr_decoder.cc:104] Required 64 get 64
I0218 14:05:34.548431 2256349 tpu_asr_model.cc:145]     wenet encoder inference start
I0218 14:05:34.562978 2256349 tpu_asr_model.cc:151]     wenet ctc inference
I0218 14:05:34.571406 2256349 asr_decoder.cc:119] forward takes 17 ms, search takes 7 ms
I0218 14:05:34.572223 2256349 asr_decoder.cc:200] Partial CTC result 甚至出现交
I0218 14:05:34.572293 2256349 websocket_server.cc:65] Partial result: [{"sentence":"甚至出现交"}]
I0218 14:05:35.085868 2256347 websocket_server.cc:90] Received 8000 samples
I0218 14:05:35.091156 2256349 asr_decoder.cc:104] Required 64 get 64
I0218 14:05:35.092988 2256349 tpu_asr_model.cc:145]     wenet encoder inference start
I0218 14:05:35.107556 2256349 tpu_asr_model.cc:151]     wenet ctc inference
I0218 14:05:35.116329 2256349 asr_decoder.cc:119] forward takes 17 ms, search takes 7 ms
I0218 14:05:35.117182 2256349 asr_decoder.cc:200] Partial CTC result 甚至出现交易几乎
I0218 14:05:35.117251 2256349 websocket_server.cc:65] Partial result: [{"sentence":"甚至出现交易几乎"}]
I0218 14:05:35.585850 2256347 websocket_server.cc:90] Received 8000 samples
I0218 14:05:35.591066 2256349 asr_decoder.cc:104] Required 64 get 64
I0218 14:05:35.592864 2256349 tpu_asr_model.cc:145]     wenet encoder inference start
I0218 14:05:35.607343 2256349 tpu_asr_model.cc:151]     wenet ctc inference
I0218 14:05:35.614553 2256349 asr_decoder.cc:119] forward takes 17 ms, search takes 5 ms
I0218 14:05:35.614774 2256349 asr_decoder.cc:200] Partial CTC result 甚至出现交易几乎停滞的
I0218 14:05:35.614790 2256349 websocket_server.cc:65] Partial result: [{"sentence":"甚至出现交易几乎停滞的"}]
I0218 14:05:36.085853 2256347 websocket_server.cc:90] Received 8000 samples
I0218 14:05:36.091145 2256349 asr_decoder.cc:104] Required 64 get 64
I0218 14:05:36.093070 2256349 tpu_asr_model.cc:145]     wenet encoder inference start
I0218 14:05:36.107636 2256349 tpu_asr_model.cc:151]     wenet ctc inference
I0218 14:05:36.116271 2256349 asr_decoder.cc:119] forward takes 17 ms, search takes 7 ms
I0218 14:05:36.117182 2256349 asr_decoder.cc:200] Partial CTC result 甚至出现交易几乎停滞的情况
I0218 14:05:36.117251 2256349 websocket_server.cc:65] Partial result: [{"sentence":"甚至出现交易几乎停滞的情况"}]
I0218 14:05:36.585665 2256347 websocket_server.cc:90] Received 3263 samples
I0218 14:05:37.044373 2256347 websocket_server.cc:212] {"signal":"end"}
I0218 14:05:37.044456 2256347 websocket_server.cc:57] Received speech end signal
I0218 14:05:37.044474 2256347 websocket_server.cc:229] Read all pcm data, wait for decoding thread
I0218 14:05:37.044581 2256349 asr_decoder.cc:104] Required 64 get 27
I0218 14:05:37.046546 2256349 tpu_asr_model.cc:145]     wenet encoder inference start
I0218 14:05:37.061100 2256349 tpu_asr_model.cc:151]     wenet ctc inference
I0218 14:05:37.068156 2256349 asr_decoder.cc:119] forward takes 17 ms, search takes 5 ms
I0218 14:05:37.068360 2256349 asr_decoder.cc:200] Partial CTC result 甚至出现交易几乎停滞的情况
I0218 14:05:37.068529 2256349 asr_decoder.cc:200] Partial CTC result 甚至出现交易几乎停滞的情况
I0218 14:05:37.068531 2256349 asr_decoder.cc:84] Rescoring cost latency: 0ms.
I0218 14:05:37.068584 2256349 websocket_server.cc:73] Final result: [{"sentence":"甚至出现交易几乎停滞的情况","word_pieces":[{"word":"甚","start":540,"end":640},{"word":"至","start":780,"end":880},{"word":"出","start":1100,"end":1200},{"word":"现","start":1340,"end":1440},{"word":"交","start":1580,"end":1680},{"word":"易","start":1820,"end":1920},{"word":"几","start":2140,"end":2240},{"word":"乎","start":2300,"end":2400},{"word":"停","start":2540,"end":2640},{"word":"滞","start":2780,"end":2880},{"word":"的","start":2940,"end":3040},{"word":"情","start":3180,"end":3280},{"word":"况","start":3420,"end":3520}]}]
```
* Step 2. Start client in another terminal
```sh
export GLOG_logtostderr=1
export GLOG_v=3
./build/bin/websocket_client_main \
    --hostname 127.0.0.1 --port 10086 \
    --wav_path ./data/test.wav \
    --continuous_decoding

I0218 14:05:32.534451 2256348 websocket_client.cc:67] {"status":"ok","type":"server_ready"}
I0218 14:05:32.536726 2256346 websocket_client_main.cc:56] Send 8000 samples
I0218 14:05:33.037812 2256346 websocket_client_main.cc:56] Send 8000 samples
I0218 14:05:33.538902 2256346 websocket_client_main.cc:56] Send 8000 samples
I0218 14:05:33.566591 2256348 websocket_client.cc:67] {"status":"ok","type":"partial_result","nbest":"[{\"sentence\":\"甚至出\"}]"}
I0218 14:05:34.039932 2256346 websocket_client_main.cc:56] Send 8000 samples
I0218 14:05:34.541009 2256346 websocket_client_main.cc:56] Send 8000 samples
I0218 14:05:34.572508 2256348 websocket_client.cc:67] {"status":"ok","type":"partial_result","nbest":"[{\"sentence\":\"甚至出现交\"}]"}
I0218 14:05:35.041317 2256346 websocket_client_main.cc:56] Send 8000 samples
I0218 14:05:35.117462 2256348 websocket_client.cc:67] {"status":"ok","type":"partial_result","nbest":"[{\"sentence\":\"甚至出现交易几乎\"}]"}
I0218 14:05:35.542326 2256346 websocket_client_main.cc:56] Send 8000 samples
I0218 14:05:35.614902 2256348 websocket_client.cc:67] {"status":"ok","type":"partial_result","nbest":"[{\"sentence\":\"甚至出现交易几乎停滞的\"}]"}
I0218 14:05:36.043318 2256346 websocket_client_main.cc:56] Send 8000 samples
I0218 14:05:36.117471 2256348 websocket_client.cc:67] {"status":"ok","type":"partial_result","nbest":"[{\"sentence\":\"甚至出现交易几乎停滞的情况\"}]"}
I0218 14:05:36.543923 2256346 websocket_client_main.cc:56] Send 3263 samples
I0218 14:05:37.068686 2256348 websocket_client.cc:67] {"status":"ok","type":"final_result","nbest":"[{\"sentence\":\"甚至出现交易几乎停滞的情况\",\"word_pieces\":[{\"word\":\"甚\",\"start\":540,\"end\":640},{\"word\":\"至\",\"start\":780,\"end\":880},{\"word\":\"出\",\"start\":1100,\"end\":1200},{\"word\":\"现\",\"start\":1340,\"end\":1440},{\"word\":\"交\",\"start\":1580,\"end\":1680},{\"word\":\"易\",\"start\":1820,\"end\":1920},{\"word\":\"几\",\"start\":2140,\"end\":2240},{\"word\":\"乎\",\"start\":2300,\"end\":2400},{\"word\":\"停\",\"start\":2540,\"end\":2640},{\"word\":\"滞\",\"start\":2780,\"end\":2880},{\"word\":\"的\",\"start\":2940,\"end\":3040},{\"word\":\"情\",\"start\":3180,\"end\":3280},{\"word\":\"况\",\"start\":3420,\"end\":3520}]}]"}
I0218 14:05:37.068723 2256348 websocket_client.cc:67] {"status":"ok","type":"speech_end"}
I0218 14:05:37.068796 2256346 websocket_client_main.cc:63] Total latency: 24ms.
```
![Client](./web/client.jpg)

### Server and web
```sh
# 1. trun on the websocket first
# 2. copy web file to local, and turn on the templates/index.html by browser
# 3. if you start server in the edge, bind ip and trun on browser in local pc
ssh -L {local-port}:127.0.0.1:{host-port} {user}@{host-ip}
```
![Web UI](./web/web.jpg)

### Hotword boosting
In Automatic Speech Recognition (ASR) systems, hotword boosting, also known as biasing or hotword replacement, is a technique that allows developers and users to specify a set of words or phrases that should be given higher priority during the speech recognition process. This technique is particularly useful for dealing with proper nouns, brand names, technical terms, or words that may not be commonly present in the standard vocabulary of the recognizer.

The purpose of hotword boosting is to increase the recognition accuracy of these specified terms, even in challenging listening conditions such as high background noise or unclear pronunciation. This is critical in many applications, such as in the medical, legal, or specialized industry sectors, where the accurate recognition of jargon is necessary.

```sh
# --context_path context.txt --context_score 10
./build/bin/decoder_main \
    --chunk_size 8 \
    --num_left_chunks 16 \
    --rescoring_weight 0.0 \
    --wav_path ./data/test.wav \
    --tpu_model_dir ./bmodel \
    --unit_path ./bmodel/units.txt \
    --continuous_decoding \
    --context_path context.txt \
    --context_score 10 2>&1 | tee log.txt

# result "停滞"  -> "停止"
test 甚至出现交易几乎停止的情况

I0219 14:34:43.942099 2511659 params.h:224] Reading SOPHGO TPU model from ./bmodel
I0219 14:34:44.577302 2511659 tpu_asr_model.cc:60] TPU Encoder:
I0219 14:34:44.577345 2511659 tpu_asr_model.cc:15]      Input-0: Shape [1,1,71,80]
I0219 14:34:44.577353 2511659 tpu_asr_model.cc:15]      Input-1: Shape [1,96,128,128]
I0219 14:34:44.577358 2511659 tpu_asr_model.cc:15]      Input-2: Shape [1,512,12,6]
I0219 14:34:44.577363 2511659 tpu_asr_model.cc:15]      Input-3: Shape [1,8,8,136]
I0219 14:34:44.577366 2511659 tpu_asr_model.cc:22]      Output-0: Shape [1,512,1,8]
I0219 14:34:44.577371 2511659 tpu_asr_model.cc:22]      Output-1: Shape [1,96,128,128]
I0219 14:34:44.577375 2511659 tpu_asr_model.cc:22]      Output-2: Shape [1,512,12,6]
I0219 14:34:44.577379 2511659 tpu_asr_model.cc:62] TPU CTC:
I0219 14:34:44.577384 2511659 tpu_asr_model.cc:15]      Input-0: Shape [1,512,1,8]
I0219 14:34:44.577389 2511659 tpu_asr_model.cc:22]      Output-0: Shape [1,5538,1,8]
I0219 14:34:44.577394 2511659 tpu_asr_model.cc:79] TPU Model Info:
I0219 14:34:44.577396 2511659 tpu_asr_model.cc:80]      chunk_size 8
I0219 14:34:44.577400 2511659 tpu_asr_model.cc:81]      num_left_chunks 16
I0219 14:34:44.577404 2511659 tpu_asr_model.cc:82]      subsampling_rate 8
I0219 14:34:44.577409 2511659 tpu_asr_model.cc:83]      right context 14
I0219 14:34:44.577412 2511659 tpu_asr_model.cc:84]      sos 5537
I0219 14:34:44.577415 2511659 tpu_asr_model.cc:85]      eos 5537
I0219 14:34:44.577419 2511659 tpu_asr_model.cc:86]      is bidirectional decoder 0
I0219 14:34:44.577423 2511659 tpu_asr_model.cc:87]      hidden_dim 512
I0219 14:34:44.577430 2511659 params.h:236] Reading unit table ./bmodel/units.txt
I0219 14:34:44.586053 2511659 params.h:261] Reading context context.txt
I0219 14:34:44.600006 2511661 decoder_main.cc:54] num frames 418
I0219 14:34:44.621109 2511661 asr_decoder.cc:104] Required 71 get 71
I0219 14:34:44.622056 2511661 tpu_asr_model.cc:145]     wenet encoder inference start
I0219 14:34:44.636265 2511661 tpu_asr_model.cc:151]     wenet ctc inference
I0219 14:34:44.645341 2511661 asr_decoder.cc:119] forward takes 16 ms, search takes 7 ms
I0219 14:34:44.646191 2511661 asr_decoder.cc:104] Required 64 get 64
I0219 14:34:44.647502 2511661 tpu_asr_model.cc:145]     wenet encoder inference start
I0219 14:34:44.661828 2511661 tpu_asr_model.cc:151]     wenet ctc inference
I0219 14:34:44.671768 2511661 asr_decoder.cc:119] forward takes 16 ms, search takes 8 ms
I0219 14:34:44.672286 2511661 asr_decoder.cc:200] Partial CTC result 甚至出
I0219 14:34:44.672300 2511661 asr_decoder.cc:212] Contexts:
I0219 14:34:44.672317 2511661 decoder_main.cc:72] Partial result: 甚至出
I0219 14:34:44.672350 2511661 asr_decoder.cc:104] Required 64 get 64
I0219 14:34:44.673466 2511661 tpu_asr_model.cc:145]     wenet encoder inference start
I0219 14:34:44.687775 2511661 tpu_asr_model.cc:151]     wenet ctc inference
I0219 14:34:44.697793 2511661 asr_decoder.cc:119] forward takes 16 ms, search takes 8 ms
I0219 14:34:44.698377 2511661 asr_decoder.cc:200] Partial CTC result 甚至出现交
I0219 14:34:44.698397 2511661 asr_decoder.cc:212] Contexts:
I0219 14:34:44.698416 2511661 decoder_main.cc:72] Partial result: 甚至出现交
I0219 14:34:44.698451 2511661 asr_decoder.cc:104] Required 64 get 64
I0219 14:34:44.699791 2511661 tpu_asr_model.cc:145]     wenet encoder inference start
I0219 14:34:44.714123 2511661 tpu_asr_model.cc:151]     wenet ctc inference
I0219 14:34:44.724177 2511661 asr_decoder.cc:119] forward takes 16 ms, search takes 8 ms
I0219 14:34:44.724795 2511661 asr_decoder.cc:200] Partial CTC result 甚至出现交易几乎
I0219 14:34:44.724822 2511661 asr_decoder.cc:212] Contexts:
I0219 14:34:44.724841 2511661 decoder_main.cc:72] Partial result: 甚至出现交易几乎
I0219 14:34:44.724876 2511661 asr_decoder.cc:104] Required 64 get 64
I0219 14:34:44.726214 2511661 tpu_asr_model.cc:145]     wenet encoder inference start
I0219 14:34:44.740545 2511661 tpu_asr_model.cc:151]     wenet ctc inference
I0219 14:34:44.751394 2511661 asr_decoder.cc:119] forward takes 16 ms, search takes 9 ms
I0219 14:34:44.751991 2511661 asr_decoder.cc:200] Partial CTC result 甚至出现交易几乎停止的
I0219 14:34:44.752027 2511661 asr_decoder.cc:212] Contexts: 停止,
I0219 14:34:44.752044 2511661 decoder_main.cc:72] Partial result: 甚至出现交易几乎停止的
I0219 14:34:44.752077 2511661 asr_decoder.cc:104] Required 64 get 64
I0219 14:34:44.753245 2511661 tpu_asr_model.cc:145]     wenet encoder inference start
I0219 14:34:44.767550 2511661 tpu_asr_model.cc:151]     wenet ctc inference
I0219 14:34:44.777616 2511661 asr_decoder.cc:119] forward takes 16 ms, search takes 8 ms
I0219 14:34:44.778268 2511661 asr_decoder.cc:200] Partial CTC result 甚至出现交易几乎停止的情况
I0219 14:34:44.778311 2511661 asr_decoder.cc:212] Contexts: 停止,
I0219 14:34:44.778329 2511661 decoder_main.cc:72] Partial result: 甚至出现交易几乎停止的情况
I0219 14:34:44.778354 2511661 asr_decoder.cc:104] Required 64 get 27
I0219 14:34:44.779408 2511661 tpu_asr_model.cc:145]     wenet encoder inference start
I0219 14:34:44.793558 2511661 tpu_asr_model.cc:151]     wenet ctc inference
I0219 14:34:44.798799 2511661 asr_decoder.cc:119] forward takes 15 ms, search takes 4 ms
I0219 14:34:44.799345 2511661 asr_decoder.cc:200] Partial CTC result 甚至出现交易几乎停止的情况
I0219 14:34:44.799384 2511661 asr_decoder.cc:212] Contexts: 停止,
I0219 14:34:44.799985 2511661 asr_decoder.cc:200] Partial CTC result 甚至出现交易几乎停止的情况
I0219 14:34:44.800017 2511661 asr_decoder.cc:212] Contexts: 停止,
I0219 14:34:44.800022 2511661 asr_decoder.cc:84] Rescoring cost latency: 0ms.
I0219 14:34:44.800026 2511661 decoder_main.cc:72] Partial result: 甚至出现交易几乎停止的情况
I0219 14:34:44.800030 2511661 decoder_main.cc:104] test Final result: 甚至出现交易几乎停止的情况
I0219 14:34:44.800037 2511661 decoder_main.cc:105] Decoded 4203ms audio taken 177ms.
[BMRT][bmcpu_setup:435] INFO:cpu_lib 'libcpuop.so' is loaded.
bmcpu init: skip cpu_user_defined
open usercpu.so, init user_cpu_init
[BMRT][BMProfile:59] INFO:Profile For arch=3
[BMRT][BMProfileDeviceBase:190] INFO:gdma=0, tiu=0, mcu=0
[BMRT][load_bmodel:1594] INFO:Loading bmodel from [./bmodel/encoder.bmodel]. Thanks for your patience...
[BMRT][load_bmodel:1503] INFO:pre net num: 0, load net num: 1
[BMRT][load_tpu_module:1575] INFO:loading firmare in bmodel
[BMRT][bmcpu_setup:435] INFO:cpu_lib 'libcpuop.so' is loaded.
bmcpu init: skip cpu_user_defined
open usercpu.so, init user_cpu_init
[BMRT][BMProfile:59] INFO:Profile For arch=3
[BMRT][BMProfileDeviceBase:190] INFO:gdma=0, tiu=0, mcu=0
[BMRT][load_bmodel:1594] INFO:Loading bmodel from [./bmodel/ctc.bmodel]. Thanks for your patience...
[BMRT][load_bmodel:1503] INFO:pre net num: 0, load net num: 1
[BMRT][load_tpu_module:1575] INFO:loading firmare in bmodel
test 甚至出现交易几乎停止的情况
I0219 14:34:44.825902 2511659 decoder_main.cc:180] Total: decoded 4203ms audio taken 177ms.
I0219 14:34:44.825933 2511659 decoder_main.cc:182] RTF: 0.04211
```