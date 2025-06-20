from langgraph.graph import StateGraph
from typing import TypedDict, Annotated
from langchain_core.runnables import RunnableConfig, Runnable
import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


class EmailInput(TypedDict):
    body: str

class EmailOutput(TypedDict):
    body: str
    intent: str


def classify_email(state: EmailInput) -> EmailOutput:
    vec = vectorizer.transform([state["body"]])
    intent = model.predict(vec)[0]
    return {"body": state["body"], "intent": intent}

def trigger_workflow(state: EmailOutput) -> EmailOutput:
    print(f"Triggering '{state['intent']}' workflow...")
    return state

graph = StateGraph(EmailInput)

graph.add_node("ClassifyEmail", classify_email)
graph.add_node("TriggerWorkflow", trigger_workflow)

graph.set_entry_point("ClassifyEmail")
graph.add_edge("ClassifyEmail", "TriggerWorkflow")
graph.set_finish_point("TriggerWorkflow")

app = graph.compile()
