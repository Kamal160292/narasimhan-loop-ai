import torch
import torch.nn as nn

# Define the AI Model
class NarasimhanLoopAI(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super(NarasimhanLoopAI, self).__init__()
        self.fc = nn.Linear(768, 1)  # Simulated Cold Spot Retrieval

    def forward(self, input_text):
        return self.fc(torch.ones((1, 768)))

# Load Model
model = NarasimhanLoopAI()
model.load_state_dict(torch.load("narasimhan_loop_ai.pth"))

# Test Retrieval
test_input = "What is Cold Spot Intelligence?"
response = model.forward(test_input)

print("🔄 AI Retrieval Test: Cold Spot Intelligence Response:", response.item())
