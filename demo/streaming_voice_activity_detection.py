"""
实时语音端点检测
"""

from typing import Any

from funasr import AutoModel

if __name__ == "__main__":
    CHUNK_SIZE = 200  # ms
    model = AutoModel(model="fsmn-vad", model_revision="v2.0.4")

    import soundfile

    wav_file: str = f"{model.model_path}/example/vad_example.wav"
    speech, sample_rate = soundfile.read(wav_file)
    chunk_stride = int(CHUNK_SIZE * sample_rate / 1000)

    cache: dict = {}
    total_chunk_num = int(len((speech) - 1) / chunk_stride + 1)
    for i in range(total_chunk_num):
        speech_chunk = speech[i * chunk_stride : (i + 1) * chunk_stride]
        is_final: bool = i == total_chunk_num - 1
        res: list[Any] = model.generate(input=speech_chunk, cache=cache, is_final=is_final, chunk_size=CHUNK_SIZE)
        if len(res[0]["value"]):
            print(res[0])
