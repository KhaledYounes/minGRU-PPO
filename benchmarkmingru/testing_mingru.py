import io
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from torch.utils.data import DataLoader, Dataset

from mingru.stacked_min_gru import StackedMinGRU

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LENGTH = 4096
BATCH_SIZE = 8
EMBED_DIM = 128
HIDDEN_DIM = 128
NUM_LAYERS = 3
NUM_EPOCHS = 10
LEARNING_RATE = 3e-4
GRAD_CLIP = 1.0
FUSED_KERNEL = True

# Load and prepare text
with io.open("Alice_in_wonderland.txt", 'r', encoding='utf-8') as f:
    text = f.read().lower()

# Train BPE tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"])
tokenizer.pre_tokenizer = Whitespace()
tokenizer.train_from_iterator([text], trainer)

# Convert text to token IDs with special tokens
encoded = tokenizer.encode("[BOS]" + text + "[EOS]").ids
vocab_size = tokenizer.get_vocab_size()

print("The model used is: minGRU")
print("Sequence length is: ", SEQ_LENGTH)
print("Use fused kernel: ", FUSED_KERNEL)
print(f"Vocabulary size: {vocab_size}")
print(f"Total tokens: {len(encoded)}")


class TextDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = torch.tensor(data, dtype=torch.long)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + 1:idx + self.seq_length + 1]
        return x, y


dataset = TextDataset(encoded, SEQ_LENGTH)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


class GRULanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.stacked_min_gru = StackedMinGRU(embed_dim, hidden_dim, num_layers, use_norm=True, use_residual=True,
                                             use_fused_kernel=FUSED_KERNEL)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x, hidden=None):
        embed = self.embed(x)
        out, hidden = self.stacked_min_gru(embed, hidden)
        logits = self.fc(out)
        return logits, hidden.squeeze(2)

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(DEVICE)


model = GRULanguageModel(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
criterion = nn.CrossEntropyLoss()


def generate_text(model, prompt, max_length=100):
    model.eval()
    input_ids = tokenizer.encode("[BOS]" + prompt).ids
    input_ids = torch.tensor(input_ids[-SEQ_LENGTH:], dtype=torch.long).unsqueeze(0).to(DEVICE)
    hidden = model.init_hidden(1)

    generated = []
    with torch.no_grad():
        for _ in range(max_length):
            logits, hidden = model(input_ids, hidden)
            logits = logits[:, -1, :]
            next_id = torch.argmax(logits, dim=-1)
            generated.append(next_id.item())
            input_ids = torch.cat([input_ids[:, 1:], next_id.unsqueeze(0)], dim=1)

    return tokenizer.decode(generated)


# Training loop
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    total_loss = 0
    start_time = time.time()

    for x, y in train_loader:
        hidden = model.init_hidden(BATCH_SIZE)

        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()

        logits, _ = model(x, hidden)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    scheduler.step(avg_loss)

    # Generate sample
    start_phrase = "first, she tried to "
    sample = generate_text(model, start_phrase, max_length=100)
    print(f"\nEpoch {epoch} | Loss: {avg_loss:.3f} | Time: {time.time() - start_time:.1f}s")
    print(f"Sample:\n{start_phrase}{sample}\n")

    torch.cuda.empty_cache()
