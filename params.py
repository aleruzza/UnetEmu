name = 'testname'
params = {
    'name': name,  
    'device': 'cuda', 
    'nepochs': 10,
    'lr': 1e-4,
    'save_model': True,
    'savedir': f'./outputs/{name}',
    'datadir': f'./data/',
    'Override': True,
    'savefreq': 20,
    'cond': True,
    'lr_decay': False,
    'resume': False,
    'sample_freq': 10, 
    'batch_size': 64,
    'rotaugm': True,
    'image_size': 128,
    'logima_freq': 20,
    'n_test_log_images': 10,
    'pretrain': False,
    'n_param' : 5,
    'n_pretrain': 10000 #note: it must be <101,000
}