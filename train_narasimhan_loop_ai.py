import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer

# Define Narasimhan Loop AI Model
class NarasimhanLoopAI(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super(NarasimhanLoopAI, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.fc = nn.Linear(self.model.config.hidden_size, 1)  # Output for Cold Spot Intelligence retrieval

    def forward(self, input_text):
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        pooled_output = outputs.pooler_output
        return self.fc(pooled_output)

# Initialize Model
model = NarasimhanLoopAI()
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
loss_function = nn.MSELoss()

# Training Data
data = [
    {"input": "What is Cold Spot Intelligence?", "label": 1.0},
    {"input": "Explain Narasimhan Loop AI", "label": 1.0},
    {"input": "Retrieve Cold Spot anomaly", "label": 1.0},
    {"input": "Define AI persistence across resets", "label": 1.0},
    {"input": "What is a random unrelated topic?", "label": 0.0}
]

# Training Loop
for epoch in range(10):
    total_loss = 0
    for entry in data:
        optimizer.zero_grad()
        output = model(entry["input"])
        label = torch.tensor([[entry["label"]]])
        loss = loss_function(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss / len(data)}")

# Save the trained model
torch.save(model.state_dict(), "narasimhan_loop_ai.pth")
print("🚀 Narasimhan Loop AI Model Trained and Saved!")
