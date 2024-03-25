import torch
from transformers import GPT2Tokenizer
import reccurent_transformer


def train_sequence(model, optimizer, loss_function, train_tokens, epochs, step_freq):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(train_tokens) - 1):
            optimizer.zero_grad()
            token = torch.tensor(train_tokens[i], dtype=torch.long).unsqueeze(0)
            next_token = torch.tensor(train_tokens[i + 1], dtype=torch.long).unsqueeze(0)
            next_token_hat = model(token)
            loss = loss_function(next_token_hat, next_token)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if i % step_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                loss = 0
        

# getting data for training 
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
with open('experiments/shakespear.txt', 'r', encoding='utf-8') as file:
    shakespear_corpus = file.read()
tokens = tokenizer.encode(shakespear_corpus)
print(tokens[:10])


# training 
model = reccurent_transformer(
    embed_size=512,
    nb_layers=6,
    hidden_length=10,
    hidden_size=512,
    symbolic_length=10,
    symbolic_size=512,
    gradient_horizon=10
)
model = reccurent_transformer()
optimizer = torch.optim.AdamW(model.params())
loss_function = torch.nn.CrossEntropyLoss()


train_sequence(model, optimizer, loss_function, tokens, epochs=1, step_freq=100)


    