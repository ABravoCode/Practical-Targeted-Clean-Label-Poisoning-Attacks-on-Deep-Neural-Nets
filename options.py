import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import argparse

def options():
    parser = argparse.ArgumentParser()

    parser.add_argument("--net", default='ResNet18', type=str, help="Net for poison generation.")
    parser.add_argument("--clean_target_id", default=886, type=int, help="Index of the target, selected from the testset")
    parser.add_argument("--clean_label", default=torch.tensor([0,]).to(device), help="Class index of the target")
    parser.add_argument("--root", default='../datasets/', help="Path of the dataset.")
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--num_poison", default=50, type=int, help="Number of the poison images")
    parser.add_argument("--alpha", default=0.1, type=float, help="Ratio of target image component.")
    parser.add_argument("--beta", default=0.75, type=float)
    parser.add_argument("--eps", default=0.03137254901, type=float, help="Clip Value.")
    parser.add_argument("--step_size", default=0.00784313725, type=float, help="LR for the inner optimizer.")
    parser.add_argument("--lr", default=1e-3, type=float, help="LR for the outter optimizer.")
    parser.add_argument("--ZO", action="store_true", help="use Zeroth-Order method to get grad.")
    
    args = parser.parse_args()

    return args
    
