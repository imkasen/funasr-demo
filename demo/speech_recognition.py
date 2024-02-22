"""
非实时语音识别
"""

from typing import Any

import gradio as gr
from funasr import AutoModel

model = AutoModel(
    model_hub="ms",  # model_hub：表示模型仓库，ms 为选择 modelscope 下载，hf 为选择 huggingface 下载。
    model="paraformer-zh",
    model_revision="v2.0.4",
    vad_model="fsmn-vad",
    vad_model_revision="v2.0.4",
    # punc_model="ct-punc-c",
    punc_model="ct-punc",
    punc_model_revision="v2.0.4",
    # spk_model="cam++",
    # spk_model_revision="v2.0.2",
)


def get_result(audio_path: str) -> str:
    """
    Recognize content from the input audio file.
    """
    # 'batch_size_s' refer to the total duration of audio file in seconds (s)
    res: list[Any] = model.generate(input=audio_path, batch_size_s=300)
    assert len(res) == 1
    print(res[0])
    res_dict: dict[str, Any] = res[0]
    assert "text" in res_dict
    return res_dict["text"]


# Graadio UI
TITLE: str = "FunASR 非实时语音识别"
with gr.Blocks(title=TITLE) as demo:
    gr.HTML(value=f"<h1 align='center'>{TITLE}</h1>")
    input_audio = gr.Audio(
        label="输入音频",
        type="filepath",
        interactive=True,
    )
    output_text = gr.Textbox(
        label="识别结果",
        lines=7,
        interactive=False,
    )

    input_audio.upload(fn=get_result, inputs=input_audio, outputs=output_text)  # pylint: disable=E1101
    input_audio.stop_recording(fn=get_result, inputs=input_audio, outputs=output_text)  # pylint: disable=E1101


if __name__ == "__main__":
    demo.queue().launch(inbrowser=True, share=False, show_api=False)
