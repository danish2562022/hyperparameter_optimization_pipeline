# Hyperparameter Optimization Pipeline for MultiGPUs

Performance of machine learning model depends on the configuration of hyperparameters. Finding the optimal hyperparameters (hypeparameter tuning), could be very time consuming and challenging.  

In this project, we automate the hyperparameter optimization pipeline for classification and regression models.



## Getting Started

To know more about Keras Tuner, kindly refer official [docs](https://keras.io/api/keras_tuner/)


### Installing


Say what the step will be

    $ git clone https://github.com/danish2562022/hyperparameter_optimization_pipeline.git
    $ cd hyperparameter_optimization_pipeline
    $ pip install -r requirements.txt



## Running

Explain how to run the automated hyperparameter optimization pipeline

### Sample Tests

    $ python .\hyperparameter_tuning_custom_training.py --config_files "config_param.json"
    
 Best model's hyperparameters get saved in best_model_params.txt


## Configuration file
Hyperparameter search space is defined in config_param.json
    
     {
    "max_number_of_layers": 5,
    "model_type" : "c",
    "min_units_per_layers" : 32,
    "max_units_per_layers" : 512,
    "num_of_classes" : 10,
    "choose_optimizer": "adam"
    }
        

## Built With

 

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code
of conduct, and the process for submitting pull requests to us.



## License



## Acknowledgments

  - Hat tip to anyone whose code is used
  - Inspiration
  - etc
