"""
实时语音识别

注：chunk_size 为流式延时配置，[0, 10, 5] 表示上屏实时出字粒度为 10*60=600ms，未来信息为 5*60=300ms。
每次推理输入为 600ms（采样点数为 16000*0.6=960），输出为对应文字，最后一个语音片段输入需要设置 is_final=True 来强制输出最后一个字。
"""

from typing import Any

import gradio as gr
import numpy as np
from funasr import AutoModel

model = AutoModel(
    model="paraformer-zh-streaming",
    model_revision="v2.0.4",
)


def get_result(audio: tuple[int, np.ndarray]):
    """
    Recognize content from the input audio file.

    :param audio: audio data
    :yield: text result
    """

    _, audio_data = audio
    chunk_size: list[int] = [0, 10, 5]  # [0, 10, 5] 600ms, [0, 8, 4] 480ms
    chunk_stride: int = chunk_size[1] * 960  # 600ms
    encoder_chunk_look_back = 4  # number of chunks to lookback for encoder self-attention
    decoder_chunk_look_back = 1  # number of encoder chunks to lookback for decoder cross-attention

    cache: dict = {}
    text_res: str = ""
    total_chunk_num = int(len(audio_data - 1) / chunk_stride + 1)
    for i in range(total_chunk_num):
        speech_chunk = audio_data[i * chunk_stride : (i + 1) * chunk_stride]
        is_final: bool = i == total_chunk_num - 1
        res: list = model.generate(
            input=speech_chunk,
            cache=cache,
            is_final=is_final,
            chunk_size=chunk_size,
            encoder_chunk_look_back=encoder_chunk_look_back,
            decoder_chunk_look_back=decoder_chunk_look_back,
        )
        assert len(res) == 1
        print(res[0])
        res_dict: dict[str, Any] = res[0]
        assert "text" in res_dict
        if res_dict["text"]:
            text_res += res_dict["text"]
            yield text_res


# Gradio UI
TITLE: str = "实时语音识别"
with gr.Blocks(title=TITLE) as demo:
    gr.HTML(f"<h1 align='center'>{TITLE}</h1>")

    input_audio = gr.Audio(label="输入音频", sources=["microphone", "upload"], type="numpy", interactive=True)
    output_text = gr.Textbox(label="识别结果", lines=7, interactive=False)

    # pylint: disable=E1101
    input_audio.stop_recording(fn=get_result, inputs=input_audio, outputs=output_text)
    input_audio.upload(fn=get_result, inputs=input_audio, outputs=output_text)


if __name__ == "__main__":
    demo.queue().launch(inbrowser=True, share=False, show_api=False)
