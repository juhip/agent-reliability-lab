import streamlit as st
from agent_reliability_lab.domains.invoice.app import build_invoice_runtime
from agent_reliability_lab.evals.runner import load_jsonl, run_cases

st.set_page_config(page_title="Agent Reliability Lab", layout="wide")
st.title("Agent Reliability Lab")
st.caption("A small harness for testing agent decisions, tool use, policy compliance, and failure modes.")

runtime = build_invoice_runtime()
left, right = st.columns(2)
with left:
    st.subheader("Reference agent: invoice exception handling")
    invoice_id = st.text_input("Invoice ID", "INV-DEMO")
    po_id = st.selectbox("PO", ["PO-1001", "PO-1002", "PO-1003", "PO-9999"])
    supplier = st.text_input("Supplier", "Northwind Components")
    amount = st.number_input("Invoice amount", value=10050.0)
    if st.button("Run agent"):
        out = runtime.run({"invoice_id":invoice_id,"po_id":po_id,"supplier":supplier,"invoice_amount":amount})
        st.json(out)
with right:
    st.subheader("Evaluation")
    if st.button("Run 20-case suite"):
        report = run_cases(runtime, load_jsonl("data/invoice_cases.jsonl"))
        st.metric("Action accuracy", f"{report['summary']['accuracy']*100:.0f}%")
        st.json(report["summary"])
        st.dataframe([{k:v for k,v in row.items() if k != "trace"} for row in report["cases"]], use_container_width=True)
