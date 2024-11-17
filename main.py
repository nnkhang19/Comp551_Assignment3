import argparse
from utils import *
from medmnist import OrganAMNIST
import torchvision.transforms as T
from torch.utils.data import DataLoader
from medmnist import INFO, Evaluator

class Normalization:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std 
    
    def __call__(self, x):
        return (x - self.mean) / self.std

def main(args):

    data_flag = 'organamnist'
    info = INFO[data_flag]

    net = create_model(args)

    transforms = T.Compose([T.ToTensor()])
    
    train_dataset = OrganAMNIST(root = "data/", split="train", size = args.img_size, transform = transforms, download = True)

    mean = np.mean([img for img, label in train_dataset], axis = 0)
    std = np.std([img for img, label in train_dataset], axis = 0)

    if args.normalized:
        transforms = T.Compose([
        T.ToTensor(),
        Normalization(mean, std), 
        T.Lambda(lambda x: x.view(-1)), ## flatten the image to a vector of 128 * 128
        AddOne(),
        ])
    else:
        transforms = T.Compose([
            T.ToTensor(), 
            T.Lambda(lambda x: x.view(-1)), 
            AddOne()
        ])

    train_dataset = OrganAMNIST(root = "data/", split="train", size = args.img_size, transform = transforms, download = True)
    val_dataset = OrganAMNIST(root = "data/", split="val", size = args.img_size, transform = transforms, download = True)
    test_dataset = OrganAMNIST(root = "data/", split="test", size = args.img_size, transform = transforms, download = True) 

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False, num_workers=args.num_workers)
   
    net, all_loss = train_model(net, train_loader, train_dataset, args)

    evaluator = Evaluator(data_flag, split = 'test', root = "data/")
    auc, acc = evaluate(net, test_loader, evaluator, split = 'test')
    
    print('Test acc: ', acc)
    print("Test auc: ", auc)

    np.savez(f"logs/results/{args.exp_name}.npz", config = args, net = net, loss = all_loss, test_acc = acc, test_auc = auc)

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    ### model configuration ###
    parser.add_argument("--num_layers", type = int, default = 1, help = "number of hidden layers")
    parser.add_argument("--hidden_dims", nargs = '+', default = [], help = 'hidden dimensions')
    parser.add_argument('--acts', nargs='+', default = [], help = "activations in hidden layers")
    parser.add_argument('--reg_type', type = str, default = None, help = 'regularization method')
    parser.add_argument("--reg_coeff", type = float, default = 0.0, help = "regularization strength")

    ### dataset and training configuration ###
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--lr", type = float, default = 1e-3, help = "learning rate")
    parser.add_argument("--batch_size", type = int, default = 512, help = "batch size")
    parser.add_argument("--num_epochs", type = int , default = 300, help = "number of training epochs")
    parser.add_argument("--num_class", type = int, default = 11, help = "number of classes")
    parser.add_argument("--num_workers", type = int, default = 6)
    parser.add_argument("--exp_name", type = str, default = "image classfication")
    parser.add_argument("--normalized", type = int, default = 1)
    parser.add_argument("--img_size", type = int, default = 28)


    args = parser.parse_args()

    # args.exp_name = f"hidden_dim_{args.hidden_dims}_activation_{args.acts}_reg_type_{args.reg_type}_reg_coeff_{args.reg_coeff}"

    main(args)
