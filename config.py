config = {
    'model': {
        'in_features': 10, 
        'num_blocks': 1, 
        'heads': 1, 
        'dropout': 0., 
        'expansion_ratio': 4, 
        'apply_activation_last': True,
        'temperature': 1
    },
    
    'ModelModule': {
        'optimizer': {
            'weight_decay': 1E-3,
            'lr': 1E-3
        },
        'lr_scheduler': {
            'gamma': 0.1,
            'milestones': [20],
        },
    },
    
    'DataModule': {
        
    },
    
    'CheckpointModule': {
        'dirpath': '/path/to/your/custom_directory/', 
        'save_top_k': -1,  # Save all checkpoints
        'every_n_epochs': 2,  # Save every 2 epochs
        'monitor': 'val_accuracy',  # Metric to monitor
        'mode': 'max',  # Save the model with the highest val_accuracy
        'filename': '{epoch}-{val_accuracy:.2f}',  # Filename format
    },
}

class Config:
    def __init__(self):
        self._config = config
        
    def get_property(self, property_name):
        if property_name not in self._config:
            return None
        
        return self._config[property_name]