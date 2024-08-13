# Mental Disorder Diagnostics Using Machine Learning from Neuroimaging Data
The FMRIAnalyser class implements all functionalities and acts as a central hub for various functionalities related to fMRI data analysis. The class encompasses both machine learning classification tasks, including mental disorder and speech-gesture prediction, as well as Searchlight Representational Analysis (RSA) to explore brain region activation patterns. Furthermore, it provides the functionality of hyperparameter tuning of autoencoder and visualization of NaNs.

## Usage
Instantiate an FMRIAnalyser object and invoke the desired method.
To execute experiments on IFG, STG, or the entire brain, simply indicate the desired lobe. The system will automatically set remaining variables. For experiments on other lobes, all variables need to be explicitly specified.