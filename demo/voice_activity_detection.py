"""
语音端点检测
"""

from typing import Any

from funasr import AutoModel

if __name__ == "__main__":
    model = AutoModel(model="fsmn-vad", model_revision="v2.0.4")
    wav_file: str = f"{model.model_path}/example/vad_example.wav"
    res: list[Any] = model.generate(input=wav_file)
    print(res[0])
