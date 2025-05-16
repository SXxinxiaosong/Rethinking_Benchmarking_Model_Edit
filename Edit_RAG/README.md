## Usage Instructions

### 1. **Replacing `model_path`**
Before running SCR, specify the path to your model. 
In the `.py` script, update the `model_path` variable to point to your model location.

### 2. **Running SCR**

You can set the following options when running SCR:  
- `model_type` can be one of `llama2`, `llama3`, `mistral`, or `llama-distill`  
- `data_type` can be `zsre`, `counterfact`, or `ELKEN`  
- `edit_scene` can be `single` or `sequential`

Example command:

```bash
python edit_rag.py --model_type llama3 --data_type ELKEN --top_k 3 --summary --edit_scene sequential
