from data_processing import  Data_p
from GPY import GPY


if __name__ == "__main__":
    # get data
    pert_data = Data_p('../autodl-tmp/data',default_pert_graph=False)
    # load dataset 
    pert_data.load(data_path='../autodl-tmp/data/norman')
    # specify data split
    pert_data.prepare_split(split='simulation', seed=1)
    # get dataloader with batch size
    pert_data.get_dataloader(batch_size=32, test_batch_size=128)

    # set up and train a model
    model0 = GYP(pert_data, device='cuda:0')
    model0.model_initialize(hidden_size=64)
    model0.train(epochs=20)

    model0.save_model('./model_result')

