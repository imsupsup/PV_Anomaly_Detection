from model import *
from dataload import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = CAE_Network_3layer(3,2,64).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args['LEARNING_RATE'])

steps = 0
total_steps = len(train_dataloader)
for epoch in range(args['NUM_EPOCH']*5):
    running_loss = 0
    for i, (X_train,_) in enumerate(train_dataloader):
        steps += 1
        X_train = X_train.to(device).float() ##
        _,output = model(X_train)
        loss = criterion(output, X_train)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()*X_train.shape[0]

        if steps % total_steps == 0:
            model.eval()

            if (epoch+1)%10==0:
                print('Epoch: {}/{}'.format(epoch+1,
                    args['NUM_EPOCH']),  'Training Loss: {:.5f}..'.format(running_loss/total_steps))
            steps = 0
            running_loss = 0
            model.train()