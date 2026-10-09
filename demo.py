import json
from agent_reliability_lab.domains.invoice.app import build_invoice_runtime

runtime = build_invoice_runtime()
example = {"invoice_id":"INV-DEMO","po_id":"PO-1001","supplier":"Northwind Components","invoice_amount":10050}
print(json.dumps(runtime.run(example, task_id="demo"), indent=2))
