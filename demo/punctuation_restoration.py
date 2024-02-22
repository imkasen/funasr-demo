"""
标点恢复
"""

from typing import Any

import gradio as gr
from funasr import AutoModel

model = AutoModel(model="ct-punc", model_revision="v2.0.4")


def get_res(text: str) -> str:
    """
    Get the result of punctuation restoration.
    """
    res: list[Any] = model.generate(input=text)
    assert len(res) == 1
    print(res[0])
    res_dict: dict[str, Any] = res[0]
    assert "text" in res_dict
    return res_dict["text"]


# Gradio UI
TITLE: str = "标点恢复"
with gr.Blocks(title=TITLE) as demo:
    gr.HTML(value=f"<h1 align='center'>{TITLE}</h1>")
    input_text = gr.Textbox(
        label="输入文本",
        value="那今天的会就到这里吧 happy new year 明年见",
        lines=3,
        interactive=True,
    )
    with gr.Row():
        clear_btn = gr.ClearButton(value="清空")
        submit_btn = gr.Button(value="提交", variant="primary")
    output_text = gr.Textbox(label="输出结果", interactive=False)

    clear_btn.add(components=[input_text, output_text])
    submit_btn.click(fn=get_res, inputs=input_text, outputs=output_text)  # pylint: disable=E1101


if __name__ == "__main__":
    demo.queue().launch(inbrowser=True, share=False, show_api=False)
