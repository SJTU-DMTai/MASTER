from master import MASTERModel
import pickle

universe = 'csi800' # or 'csi800'

# Please install qlib first before load the data.
with open(f'data/{universe}/{universe}_dl_train.pkl', 'rb') as f:
    dl_train = pickle.load(f)
with open(f'data/{universe}/{universe}_dl_valid.pkl', 'rb') as f:
    dl_valid = pickle.load(f)
with open(f'data/{universe}/{universe}_dl_test.pkl', 'rb') as f:
    dl_test = pickle.load(f)
print("Data Loaded.")

d_feat = 158
d_model = 256
t_nhead = 4
s_nhead = 2
dropout = 0.5
gate_input_start_index=158
gate_input_end_index = 221

if universe == 'csi300':
    beta = 10
elif universe == 'csi800':
    beta = 5

n_epoch = 40
lr = 8e-6
GPU = 0
seed = 0
train_stop_loss_thred = 0.95

model = MASTERModel(
    d_feat = d_feat, d_model = d_model, t_nhead = t_nhead, s_nhead = s_nhead, T_dropout_rate=dropout, S_dropout_rate=dropout,
    beta=beta, gate_input_end_index=gate_input_end_index, gate_input_start_index=gate_input_start_index,
    n_epochs=n_epoch, lr = lr, GPU = GPU, seed = seed, train_stop_loss_thred = train_stop_loss_thred,
    save_path='model/', save_prefix=universe
)

# Train
model.fit(dl_train, dl_valid)
print("Model Trained.")

# Test
predictions, metrics = model.predict(dl_test)
print(metrics)

# Load and Test
#param_path = f'model/{universe}master_0.pkl.'
#print(f'Model Loaded from {param_path}')
#model.load_param(param_path)
#predictions, metrics = model.predict(dl_test)
#print(metrics)



