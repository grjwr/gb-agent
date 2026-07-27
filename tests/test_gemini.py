import sys; sys.path.insert(0, "/home/akumar/gb-agent")
from agent.arbiter_gemini import GeminiArbiter
a = GeminiArbiter()
ev_true = [{"text": "The Senate passed the infrastructure bill 68-32 on Tuesday and the president signed it into law that afternoon."}]
print("TRUE:", a.verdict("The president signed an infrastructure bill into law on Tuesday.", ev_true))
ev_false = [{"text": "The moon is composed primarily of silicate rock and metal, with no dairy content."}]
print("FALSE:", a.verdict("The moon is made entirely of compressed cheese.", ev_false))
print("UNVERIF:", a.verdict("A local council approved a road budget.", []))
