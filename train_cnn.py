import argparse
import torch 
from utils import *
from medmnist import OrganAMNIST
import torchvision.transforms as T
from torch.utils.data import DataLoader
from cnn import CNN, Resnet
from medmnist import INFO, Evaluator
import random

def set_seed(seed):
    # Set the seed for generating random numbers
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # If using CUDA
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    
    # Ensure that operations using randomness are deterministic on the GPU for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set a fixed seed value

class Normalization:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std 
    
    def __call__(self, x):
        return (x - self.mean) / self.std

def main(args):
  
    set_seed(args.seed)

    data_flag = 'organamnist'
    info = INFO[data_flag]
    print(info)

    if args.use_pretrained:
        net = Resnet(args).cuda()
    else:
        net = CNN(args.in_channels, args.num_class, args)
        net = net.cuda()

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr = args.lr)
    
    transforms = T.Compose([T.ToTensor()])
    
    train_dataset = OrganAMNIST(root = "data/", split="train", size = args.img_size, transform = transforms, download = True)

    mean = np.mean([img for img, label in train_dataset], axis = 0)
    std = np.std([img for img, label in train_dataset], axis = 0)

    if args.use_pretrained:
        transforms = T.Compose([
        T.ToTensor(),
        Normalization(mean = mean, std = std),
        #T.Resize((128, 128), antialias=True), 
        T.Lambda(lambda x: x.repeat(3, 1, 1))
        ])
    else:
        transforms = T.Compose([
        T.ToTensor(),
        Normalization(mean = mean, std =std)
        ])

    train_dataset = OrganAMNIST(root = "data/", split="train", size = args.img_size, transform = transforms, download = True)
    val_dataset = OrganAMNIST(root = "data/", split="val", size = args.img_size, transform = transforms, download = True)
    test_dataset = OrganAMNIST(root = "data/", split="test", size = args.img_size, transform = transforms, download = True)

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False, num_workers=args.num_workers)
   
    net, all_loss = train_cnn(net, loss_fn, optimizer, train_loader, train_dataset, args)

    evaluator = Evaluator(data_flag, split = 'test', root = "data/")
    auc, acc = evaluate_cnn(net, test_loader, evaluator, split = 'test')

    print('Test accuracy: ', acc)
    print("Test auc: ", auc)

    np.savez(f"logs/results/{args.exp_name}.npz", config = args, loss = all_loss, test_accuracy = acc, auc = auc)

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    ### model configuration ###
    parser.add_argument("--in_channels", nargs = '+', default = [], help = 'cnn channels')
    parser.add_argument("--hidden_dim", type = int, default = 256, help = 'fc hidden dimension')
    parser.add_argument("--pool", type = int, default = 0)

    ### dataset and training configuration ###
    parser.add_argument("--img_size", type = int, default = 28)
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--lr", type = float, default = 1e-3, help = "learning rate")
    parser.add_argument("--batch_size", type = int, default = 512, help = "batch size")
    parser.add_argument("--num_epochs", type = int , default = 300, help = "number of training epochs")
    parser.add_argument("--num_class", type = int, default = 11, help = "number of classes")
    parser.add_argument("--num_workers", type = int, default = 6)
    parser.add_argument("--exp_name", type = str, default = "image classfication")
    parser.add_argument("--use_pretrained", type = int, default = 1)


    args = parser.parse_args()

    # args.exp_name = f"hidden_dim_{args.hidden_dims}_activation_{args.acts}_reg_type_{args.reg_type}_reg_coeff_{args.reg_coeff}"

    main(args)