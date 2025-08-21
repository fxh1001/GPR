# Incorporating Valuable Prior Knowledge to Improve Deep Learning Prediction of Genetic Perturbation Responses

This is a method that can predict transcriptional response to both single and multi-gene perturbations using single-cell RNA-sequencing data from perturbational screens. 


<p align="center"><img src="https://github.com/fxh1001/GPR/blob/main/image/perturb_fig1.png"  width="900px" /></p>


### Install the required dependencies
```python
pip install -r requirements.txt
```



### Core API Interface

This is an example implementation of our model.

```python
from data_processing import  Data_p
from GPY import GPY

# get data
pert_data = Data_p('./data',default_pert_graph=False)
# load dataset 
pert_data.load(data_path='./data/norman')
# specify data split
pert_data.prepare_split(split='simulation', seed=1)
# get dataloader with batch size
pert_data.get_dataloader(batch_size=32, test_batch_size=128)

# set up and train a model
model0 = GYP(pert_data, device='cuda:0')
model0.model_initialize(hidden_size=64)
model0.train(epochs=20)

model0.save_model('./model_result')


model0.load_pretrained('./model_result')

# predict
model0.predict([['CBL', 'CNN1'], ['FEV']])
model0.GI_predict(['CBL', 'CNN1'], GI_genes_file="./genes_with_hi_mean.npy")
```
