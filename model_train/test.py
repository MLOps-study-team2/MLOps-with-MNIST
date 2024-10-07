# model_train/test.py
import torch

def test(model, device, loss_fn, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            test_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    average_loss = test_loss / total
    test_acc = correct / total
    print(f'\nTest set: Average loss: {average_loss:.4f}, Accuracy: {correct}/{total}'
          f' ({100. * test_acc:.0f}%)\n')
    
    return average_loss, test_acc
