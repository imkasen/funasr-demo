"""
时间戳预测
"""

from typing import Any

from funasr import AutoModel

if __name__ == "__main__":
    model = AutoModel(model="fa-zh", model_revision="v2.0.4")
    wav_file: str = f"{model.model_path}/example/asr_example.wav"
    text_file: str = f"{model.model_path}/example/text.txt"
    res: list[Any] = model.generate(input=(wav_file, text_file), data_type=("sound", "text"))
    print(res[0])
