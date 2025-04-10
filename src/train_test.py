import torch
import torch.nn as nn


def train_model(model, train_dataloader, valid_dataloader, criterion, optimizer, device, epochs):
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_total_loss = 0
        for input_ids, target_ids in train_dataloader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            loss.backward()
            optimizer.step()

            train_total_loss += loss.item()

        train_avg_loss = train_total_loss / len(train_dataloader)
        train_ppl = torch.exp(torch.tensor(train_avg_loss))

        # 验证阶段
        model.eval()
        valid_total_loss = 0
        with torch.no_grad():
            for input_ids, target_ids in valid_dataloader:
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)

                logits = model(input_ids)
                loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                valid_total_loss += loss.item()

        valid_avg_loss = valid_total_loss / len(valid_dataloader)
        valid_ppl = torch.exp(torch.tensor(valid_avg_loss))

        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_avg_loss}, Train PPL: {train_ppl}, Valid Loss: {valid_avg_loss}, Valid PPL: {valid_ppl}')


def test_model(model, test_dataloader, criterion, device):
    model.eval()
    test_total_loss = 0
    with torch.no_grad():
        for input_ids, target_ids in test_dataloader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            test_total_loss += loss.item()

    test_avg_loss = test_total_loss / len(test_dataloader)
    test_ppl = torch.exp(torch.tensor(test_avg_loss))
    print(f'Test Loss: {test_avg_loss}, Test PPL: {test_ppl}')
