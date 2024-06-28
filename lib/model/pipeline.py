from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from peft.peft_model import PeftModel
from peft.config import PeftConfig

def get_pipeline(model_name: str) -> HuggingFacePipeline:
    model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", trust_remote_code=False, revision="main"
    )

    config = PeftConfig.from_pretrained(
        "SatvikG7/OpenNEP-Mistral-7B-Instruct-v0.2-GPTQ-ft"
    )
    model = PeftModel.from_pretrained(
        model, "SatvikG7/OpenNEP-Mistral-7B-Instruct-v0.2-GPTQ-ft"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)  # type: ignore
    hf = HuggingFacePipeline(pipeline=pipe)
    return hf
