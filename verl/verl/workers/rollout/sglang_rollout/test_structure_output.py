import sglang as sgl
from sglang.utils import wait_for_server, print_highlight, terminate_process
from multiprocessing import freeze_support  # only needed on Windows
import asyncio

async def main():
    # 1) launch engine inside async context
    llm = sgl.Engine(
        model_path="Qwen/Qwen2.5-VL-3B-Instruct",
        grammar_backend="xgrammar"
    )
    # 2) wait until the LLM subprocesses are ready
    wait_for_server()

    prompts = [
        "Please provide information about London as a major global city:",
        "Please provide information about Paris as a major global city:",
    ]
    sampling_params = {
        "temperature": 0.8,
        "top_p": 0.95,
        "regex": "(France|England)",
    }

    # 3) await the async_generate call, note the comma after sampling_params
    outputs = await llm.async_generate(
        prompts,
        sampling_params=sampling_params,
        return_logprob=True,
    )

    for prompt, output in zip(prompts, outputs):
        print_highlight("===============================")
        print_highlight(f"Prompt: {prompt}\nGenerated text: {output['text']}")

    # 4) clean up
    terminate_process(llm)

if __name__ == "__main__":
    freeze_support()       # safe on Linux, necessary on Windows
    asyncio.run(main())
