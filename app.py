import gradio as gr
import os
import json
from functools import partial

# READ_ONLY=True
READ_ONLY=False

# DATASET_NAME = "2wikimqa"
DATASET_NAME = "hotpotqa"

image_paths = [
    f"visualizations/{DATASET_NAME}/Llama-3.1-8B",
    # f"visualizations/{DATASET_NAME}/Llama-3.1-8B-MuDAF",
    f"visualizations/{DATASET_NAME}/Llama-3.1-8B-Vanilla-SFT",
]


css_style = """
.button_row {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-around;
}

.passage_button {
    flex: 1 1 4%;
    margin: 2px;
    padding: 2px;
    min-width: 20px !important;  
    max-width: 40px;  
    background-color: transparent;
}

.scroll1{
    overflow-x: auto;
    white-space: nowrap;
}
"""

with open(f"{DATASET_NAME}_outputs.json", 'r') as f:
    question_info = json.load(f)
with open(f"{DATASET_NAME}_passages.json", 'r') as f:
    passages = json.load(f)
now_passages = []
if os.path.exists(f"{DATASET_NAME}_golden_passages_ids.json"):
    with open(f"{DATASET_NAME}_golden_passages_ids.json", 'r') as f:
        golden_passages_ids = json.load(f)
else:
    golden_passages_ids = [[0,0, ""] for _ in range(len(question_info))]
def get_info(now_index):
    global now_passages
    images = []
    texts = []
    info = None
    preds = []
    now_passages = passages[now_index]
    for path in image_paths:
        dir = os.path.join(path, str(now_index))
        if os.path.exists(dir): 
            image_files = [f for f in os.listdir(dir) if f.endswith('.png')]
            if image_files:
                image_file = image_files[0]
                images.append(os.path.join(dir, image_file))
                with open(os.path.join(dir, "question_info.json"), "r") as f:
                    pred_info = json.load(f)
                
                item_info = question_info[now_index]
                question = item_info["question"]
                answer = item_info["answer"]
                length = pred_info["length"]
                pred = pred_info["pred"]
                preds.append(pred)
                passage_titles = [item.strip() for item in item_info["passages"]]
                passage_title = '\n'.join(passage_titles)
                shown_info = f"Question: {question}\nAnswer: {answer}\nLength: {length}\nPassage Titles:\n{passage_title}"
                if info is None:
                    info = shown_info
            else:
                images.append(None)
        else:
            images.append(None) 
            texts.append("No image available")
    if len(preds)>0:
        pred_info = '\n\n'.join(preds)
        info += f"\n\nPredictions:\n{pred_info}"
    return images, info

def jump_to_item(now_index, next_index, golden_id1, golden_id2, comment):
    if now_index >= 0 and now_index < len(question_info):
        golden_passages_ids[now_index] = [golden_id1, golden_id2, comment]
        print(golden_passages_ids[now_index])
        
    images, texts = get_info(next_index)
    return (*images, texts, golden_passages_ids[next_index][0], golden_passages_ids[next_index][1], golden_passages_ids[next_index][2], next_index, next_index)

def update(state_index, golden_id1, golden_id2, comment, direction):
    if direction == -1 and state_index <= 0:
        next_index = len(question_info) - 1
    elif direction == 1 and state_index >= len(question_info) - 1:
        next_index = 0
    else:
        next_index = state_index + direction
    return jump_to_item(state_index, next_index, golden_id1, golden_id2, comment)

def on_save(now_index, golden_id1, golden_id2, comment):
    if now_index >= 0 and now_index < len(question_info):
        golden_passages_ids[now_index] = [golden_id1, golden_id2, comment]
        
    with open(f"{DATASET_NAME}_golden_passages_ids.json", 'w') as f:
        json.dump(golden_passages_ids, f, ensure_ascii=False, indent=4)


try:
    with gr.Blocks(css=css_style) as demo:
        now_index = gr.State(-1)
        with gr.Row():
            jump_to = gr.Number(label="Jump to", minimum=0, maximum=len(question_info)-1, interactive=True)
            save_button = gr.Button("Save", interactive=(not READ_ONLY))
        
        with gr.Row():
            image_outputs = [gr.Image(label=f"Image {os.path.basename(image_paths[i])}") for i in range(len(image_paths))]
            
        with gr.Row():
            golden_id1 = gr.Number(label="Golden ID 1", interactive = (not READ_ONLY))
            golden_id2 = gr.Number(label="Golden ID 2", interactive = (not READ_ONLY))
        comment = gr.Textbox(label="Comment", interactive=True)
        
        text_outputs = gr.Textbox(label="Text", interactive=False, elem_classes="scroll1")
        with gr.Row(elem_classes="button_row"):
            buttons = [gr.Button(f"{i+1}", elem_classes="passage_button") for i in range(16)]
        modal = gr.Textbox(visible=False)
        
        with gr.Row():
            prev_button = gr.Button("Previous Images")
            next_button = gr.Button("Next Images")
        
        def button_click(index):
            if index < len(now_passages):
                return gr.update(value=now_passages[index], visible=True)
            else:
                return gr.update(value=None, visible=False)
        
        for i, button in enumerate(buttons):
            button.click(fn=partial(button_click, index=i), outputs=[modal])
            

        jump_to.submit(fn=jump_to_item, inputs=[now_index, jump_to, golden_id1, golden_id2, comment], outputs=[*image_outputs, text_outputs, golden_id1, golden_id2, comment, now_index, jump_to])
        save_button.click(fn=on_save, inputs=[now_index, golden_id1, golden_id2, comment])
        prev_button.click(fn=partial(update, direction = -1), inputs=[now_index, golden_id1, golden_id2, comment], outputs=[*image_outputs, text_outputs, golden_id1, golden_id2, comment, now_index, jump_to])
        next_button.click(fn=partial(update, direction = 1), inputs=[now_index, golden_id1, golden_id2, comment], outputs=[*image_outputs, text_outputs, golden_id1, golden_id2, comment, now_index, jump_to])
    demo.launch()
except Exception as e:
    print(e)
    import pdb; pdb.set_trace()

if not READ_ONLY:
    with open(f"{DATASET_NAME}_golden_passages_ids.json", 'w') as f:
        json.dump(golden_passages_ids, f, ensure_ascii=False, indent=4)