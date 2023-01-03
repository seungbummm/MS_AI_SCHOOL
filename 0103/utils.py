import matplotlib.pyplot as plt
import glob
import torch
import os

def data_size_show(data_dir):
    x_plt = []  # 폴더명
    y_plt = []  # 폴더 별 크기

    # 데이터 분포 보기

    for directory in os.listdir(data_dir):
        # print(directory)
        x_plt.append(directory)
        # print(len(os.listdir(os.path.join(data_dir, directory))))
        y_plt.append(len(os.listdir(os.path.join(data_dir, directory))))
        

    # creating the bar plot
    fig, ax = plt.subplots(figsize=(16,16))
    plt.barh(x_plt, y_plt, color = 'maroon')
    # remove x, y Ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    # add padding between axes and labels
    ax.xaxis.set_tick_params(pad = 5)
    ax.yaxis.set_tick_params(pad = 10)

    # show top values
    ax.invert_yaxis()

    plt.ylabel('Bark Type')
    plt.xlabel('No. of image')
    plt.title('Bark Texture Dataset')

    plt.show()

def train(num_epoch, model, train_loader, val_loader, criterion, optimizer,
          save_dir, device):
    print("Start training.....")
    total = 0
    best_loss = 9999

    for epoch in range(num_epoch) :
        for i , (imgs, labels) in enumerate(train_loader) :
            img , label = imgs.to(device) , labels.to(device)
            output = model(img)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _,argmax = torch.max(output, 1)
            acc = (label == argmax).float().mean()

            total += label.size(0)

            if (i+1) % 10 == 0 :
                print("Epoch>> [{}/{}], step>> [{}/{}], Loss>> {:.4f}, acc>> "
                      "{:.2f}%".format(
                        epoch + 1,
                        num_epoch,
                        i + 1,
                        len(train_loader),
                        loss.item(),
                        acc.item() * 100
                ))
        avrg_loss, val_acc = validation(model, val_loader, criterion, device)
        if avrg_loss < best_loss :
            print("Best pt save")
            best_loss = avrg_loss
            save_model(model, save_dir)

    save_model(model, save_dir, file_name="last.pt")

def validation(model, val_loader, criterion, device) :
    print("val Start !!! ")
    model.eval()
    with torch.no_grad() :
        total = 0
        correct = 0
        total_loss = 0
        cnt = 0
        batch_loss = 0

        for i, (imgs, labels) in enumerate(val_loader) :
            imgs, labels = imgs.to(device) , labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            batch_loss += loss.item()

            total += imgs.size(0)
            _, argmax = torch.max(outputs, 1)
            correct += (labels == argmax).sum().item()
            total_loss += loss.item()
            cnt += 1

    avrg_loss = total_loss / cnt
    val_acc = (correct / total * 100)
    print("Acc >> {:.2f} Average loss >> {:.4f}".format(
        val_acc,
        avrg_loss
    ))

    model.train()

    return avrg_loss, val_acc


def save_model(model, save_dir, file_name ="best.pt"):
    output_path = os.path.join(save_dir, file_name)
    torch.save(model.state_dict(), output_path)