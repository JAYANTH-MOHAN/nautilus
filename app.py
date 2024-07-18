from vllm import LLM, SamplingParams
from huggingface_hub import login
login("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")

prompts= "Suggest some ways to reduce gun violence"
sampling_params = SamplingParams(temperature=0.8,top_p=0.5,top_k=-1,min_p=0.4,logprobs=1,seed=42,max_tokens=500)
llm = LLM(model="meta-llama/Meta-Llama-3-8B",gpu_memory_utilization=0.8,max_model_len=4096)

outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")