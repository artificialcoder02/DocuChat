from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline
import torch

model = "tiiuae/falcon-7b-instruct" #tiiuae/falcon-40b-instruct

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = pipeline(
    "text-generation", #task
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length=10000,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})

from langchain import PromptTemplate,  LLMChain

template = """
I want you to act as a SRS documentation expert. I will be providing you the snippets of the software documentation and you need to make sure that: “The documents should have a high quality of completion content, such that it is accurate, relevant, coherent, logical, and informative. The documents should be proofread and edited for grammar, spelling, punctuation, and style errors.”
Question: {question}
Answer:"""
prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "Regenerate this snippet and implement the required changes: This application is a platform for data scientists to create, train, and deploy machine learning models in one place. It allows you to where  1). Start a project and invite other collaborators to work on it. 2).Read data from various sources and explore it using tools like jupyter notebook. 3). Preprocess the data and perform feature engineering to prepare it for modeling. 4). Choose from different machine learning algorithms and train a model on the data. 5). Validate the model using various metrics and techniques. 6).Deploy the model as an API or a web app using MLOps practices. This application aims to simplify and streamline the machine learning workflow and make it accessible to data scientists of different skill levels."

print(llm_chain.run(question))