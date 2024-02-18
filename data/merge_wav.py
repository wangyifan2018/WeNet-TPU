import wave
import os

def combine_wav_files(folder_path, output_file):
    files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    files.sort()  # 可以根据需要调整排序方式

    with wave.open(output_file, 'wb') as outfile:
        # 加载第一个文件，以获取参数
        with wave.open(os.path.join(folder_path, files[0]), 'rb') as infile:
            params = infile.getparams()
            outfile.setparams(params)

        # 现在将所有文件的帧合并到输出文件中
        for file in files:
            with wave.open(os.path.join(folder_path, file), 'rb') as infile:
                while True:
                    frames = infile.readframes(65536)
                    if not frames:
                        break
                    outfile.writeframes(frames)

# 设置源文件夹和目标WAV文件名
source_folder = 'source_file'
output_wav = 'combined_output.wav'

combine_wav_files(source_folder, output_wav)
