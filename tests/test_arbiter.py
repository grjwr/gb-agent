import sys; sys.path.insert(0, "/home/akumar/gb-agent")
from agent.arbiter import Arbiter
a = Arbiter()
print("model loaded\n", flush=True)
ev_true = [{"text": "The Senate passed the infrastructure bill 68-32 on Tuesday and the president signed it into law that afternoon."}]
print("TRUE case:", a.verdict("The president signed an infrastructure bill into law on Tuesday.", ev_true), flush=True)
ev_false = [{"text": "The moon is composed primarily of silicate rock and metal, with no dairy content whatsoever."}]
print("FALSE case:", a.verdict("The moon is made entirely of compressed cheese.", ev_false), flush=True)
print("UNVERIFIABLE case:", a.verdict("A local council approved a road budget.", []), flush=True)
