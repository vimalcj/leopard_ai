# This file has been autogenerated by version 1.54.0 of the Azure Automated Machine Learning SDK.


import numpy
import numpy as np
import pandas as pd
import pickle
import argparse


# For information on AzureML packages: https://docs.microsoft.com/en-us/python/api/?view=azure-ml-py
from azureml.training.tabular._diagnostics import logging_utilities


def setup_instrumentation(automl_run_id):
    import logging
    import sys

    from azureml.core import Run
    from azureml.telemetry import INSTRUMENTATION_KEY, get_telemetry_log_handler
    from azureml.telemetry._telemetry_formatter import ExceptionFormatter

    logger = logging.getLogger("azureml.training.tabular")

    try:
        logger.setLevel(logging.INFO)

        # Add logging to STDOUT
        stdout_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stdout_handler)

        # Add telemetry logging with formatter to strip identifying info
        telemetry_handler = get_telemetry_log_handler(
            instrumentation_key=INSTRUMENTATION_KEY, component_name="azureml.training.tabular"
        )
        telemetry_handler.setFormatter(ExceptionFormatter())
        logger.addHandler(telemetry_handler)

        # Attach run IDs to logging info for correlation if running inside AzureML
        try:
            run = Run.get_context()
            return logging.LoggerAdapter(logger, extra={
                "properties": {
                    "codegen_run_id": run.id,
                    "automl_run_id": automl_run_id
                }
            })
        except Exception:
            pass
    except Exception:
        pass

    return logger


automl_run_id = 'leopard_precision_48'
logger = setup_instrumentation(automl_run_id)


def split_dataset(X, y, weights, split_ratio, should_stratify):
    '''
    Splits the dataset into a training and testing set.

    Splits the dataset using the given split ratio. The default ratio given is 0.25 but can be
    changed in the main function. If should_stratify is true the data will be split in a stratified
    way, meaning that each new set will have the same distribution of the target value as the
    original dataset. should_stratify is true for a classification run, false otherwise.
    '''
    from sklearn.model_selection import train_test_split

    random_state = 42
    if should_stratify:
        stratify = y
    else:
        stratify = None

    if weights is not None:
        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
            X, y, weights, stratify=stratify, test_size=split_ratio, random_state=random_state
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=stratify, test_size=split_ratio, random_state=random_state
        )
        weights_train, weights_test = None, None

    return (X_train, y_train, weights_train), (X_test, y_test, weights_test)


def get_training_dataset(dataset_uri):
    
    from azureml.core.run import Run
    from azureml.data.abstract_dataset import AbstractDataset
    
    logger.info("Running get_training_dataset")
    ws = Run.get_context().experiment.workspace
    dataset = AbstractDataset._load(dataset_uri, ws)
    return dataset.to_pandas_dataframe()


def prepare_data(dataframe):
    '''
    Prepares data for training.
    
    Cleans the data, splits out the feature and sample weight columns and prepares the data for use in training.
    This function can vary depending on the type of dataset and the experiment task type: classification,
    regression, or time-series forecasting.
    '''
    
    from azureml.training.tabular.preprocessing import data_cleaning
    
    logger.info("Running prepare_data")
    label_column_name = 'PRECISION'
    
    # extract the features, target and sample weight arrays
    y = dataframe[label_column_name].values
    X = dataframe.drop([label_column_name], axis=1)
    sample_weights = None
    X, y, sample_weights = data_cleaning._remove_nan_rows_in_X_y(X, y, sample_weights,
     is_timeseries=False, target_column=label_column_name)
    
    return X, y, sample_weights


def get_mapper_0(column_names):
    from sklearn.impute import SimpleImputer
    from sklearn_pandas.dataframe_mapper import DataFrameMapper
    from sklearn_pandas.features_generator import gen_features
    
    definition = gen_features(
        columns=column_names,
        classes=[
            {
                'class': SimpleImputer,
                'add_indicator': False,
                'copy': True,
                'fill_value': None,
                'missing_values': numpy.nan,
                'strategy': 'mean',
                'verbose': 0,
            },
        ]
    )
    mapper = DataFrameMapper(features=definition, input_df=True, sparse=True)
    
    return mapper
    
    
def generate_data_transformation_config():
    '''
    Specifies the featurization step in the final scikit-learn pipeline.
    
    If you have many columns that need to have the same featurization/transformation applied (for example,
    50 columns in several column groups), these columns are handled by grouping based on type. Each column
    group then has a unique mapper applied to all columns in the group.
    '''
    from sklearn.pipeline import FeatureUnion
    
    column_group_0 = [['chroma_stft'], ['rms'], ['spectral_centroid'], ['spectral_bandwidth'], ['rolloff'], ['zero_crossing_rate'], ['mfcc1'], ['mfcc2'], ['mfcc3'], ['mfcc4'], ['mfcc5'], ['mfcc6'], ['mfcc7'], ['mfcc8'], ['mfcc9'], ['mfcc10'], ['mfcc11'], ['mfcc12'], ['mfcc13'], ['mfcc14'], ['mfcc15'], ['mfcc16'], ['mfcc17'], ['mfcc18'], ['mfcc19'], ['mfcc20']]
    
    mapper = get_mapper_0(column_group_0)
    return mapper
    
    
def generate_preprocessor_config_0():
    '''
    Specifies a preprocessing step to be done after featurization in the final scikit-learn pipeline.
    
    Normally, this preprocessing step only consists of data standardization/normalization that is
    accomplished with sklearn.preprocessing. Automated ML only specifies a preprocessing step for
    non-ensemble classification and regression models.
    '''
    from sklearn.preprocessing import MinMaxScaler
    
    preproc = MinMaxScaler(
        clip=False,
        copy=True,
        feature_range=(0, 1)
    )
    
    return preproc
    
    
def generate_algorithm_config_0():
    from sklearn.ensemble import ExtraTreesRegressor
    
    algorithm = ExtraTreesRegressor(
        bootstrap=True,
        ccp_alpha=0.0,
        criterion='mse',
        max_depth=None,
        max_features=0.5,
        max_leaf_nodes=None,
        max_samples=None,
        min_impurity_decrease=0.0,
        min_samples_leaf=0.01091729022778783,
        min_samples_split=0.002602463309528381,
        min_weight_fraction_leaf=0.0,
        n_estimators=200,
        n_jobs=-1,
        oob_score=False,
        random_state=None,
        verbose=0,
        warm_start=False
    )
    
    return algorithm
    
    
def generate_preprocessor_config_1():
    from sklearn.preprocessing import StandardScaler
    
    preproc = StandardScaler(
        copy=True,
        with_mean=True,
        with_std=True
    )
    
    return preproc
    
    
def generate_algorithm_config_1():
    from sklearn.ensemble import ExtraTreesRegressor
    
    algorithm = ExtraTreesRegressor(
        bootstrap=True,
        ccp_alpha=0.0,
        criterion='mse',
        max_depth=None,
        max_features=0.6,
        max_leaf_nodes=None,
        max_samples=None,
        min_impurity_decrease=0.0,
        min_samples_leaf=0.023457724724161487,
        min_samples_split=0.015297321160913582,
        min_weight_fraction_leaf=0.0,
        n_estimators=400,
        n_jobs=-1,
        oob_score=False,
        random_state=None,
        verbose=0,
        warm_start=False
    )
    
    return algorithm
    
    
def generate_preprocessor_config_2():
    from sklearn.preprocessing import StandardScaler
    
    preproc = StandardScaler(
        copy=True,
        with_mean=True,
        with_std=True
    )
    
    return preproc
    
    
def generate_algorithm_config_2():
    from sklearn.ensemble import ExtraTreesRegressor
    
    algorithm = ExtraTreesRegressor(
        bootstrap=True,
        ccp_alpha=0.0,
        criterion='mse',
        max_depth=None,
        max_features=0.5,
        max_leaf_nodes=None,
        max_samples=None,
        min_impurity_decrease=0.0,
        min_samples_leaf=0.01321775243133542,
        min_samples_split=0.012814223889440833,
        min_weight_fraction_leaf=0.0,
        n_estimators=400,
        n_jobs=-1,
        oob_score=False,
        random_state=None,
        verbose=0,
        warm_start=False
    )
    
    return algorithm
    
    
def generate_preprocessor_config_3():
    from sklearn.preprocessing import StandardScaler
    
    preproc = StandardScaler(
        copy=True,
        with_mean=True,
        with_std=True
    )
    
    return preproc
    
    
def generate_algorithm_config_3():
    from sklearn.ensemble import ExtraTreesRegressor
    
    algorithm = ExtraTreesRegressor(
        bootstrap=False,
        ccp_alpha=0.0,
        criterion='mse',
        max_depth=None,
        max_features=0.7,
        max_leaf_nodes=None,
        max_samples=None,
        min_impurity_decrease=0.0,
        min_samples_leaf=0.03438518777954402,
        min_samples_split=0.0018261584682702607,
        min_weight_fraction_leaf=0.0,
        n_estimators=50,
        n_jobs=-1,
        oob_score=False,
        random_state=None,
        verbose=0,
        warm_start=False
    )
    
    return algorithm
    
    
def generate_preprocessor_config_4():
    from sklearn.preprocessing import RobustScaler
    
    preproc = RobustScaler(
        copy=True,
        quantile_range=[10, 90],
        unit_variance=False,
        with_centering=False,
        with_scaling=True
    )
    
    return preproc
    
    
def generate_algorithm_config_4():
    from sklearn.ensemble import ExtraTreesRegressor
    
    algorithm = ExtraTreesRegressor(
        bootstrap=True,
        ccp_alpha=0.0,
        criterion='mse',
        max_depth=None,
        max_features=0.6,
        max_leaf_nodes=None,
        max_samples=None,
        min_impurity_decrease=0.0,
        min_samples_leaf=0.01600296187889058,
        min_samples_split=0.0630957344480193,
        min_weight_fraction_leaf=0.0,
        n_estimators=200,
        n_jobs=-1,
        oob_score=False,
        random_state=None,
        verbose=0,
        warm_start=False
    )
    
    return algorithm
    
    
def generate_preprocessor_config_5():
    from sklearn.preprocessing import StandardScaler
    
    preproc = StandardScaler(
        copy=True,
        with_mean=True,
        with_std=True
    )
    
    return preproc
    
    
def generate_algorithm_config_5():
    from sklearn.ensemble import RandomForestRegressor
    
    algorithm = RandomForestRegressor(
        bootstrap=True,
        ccp_alpha=0.0,
        criterion='mse',
        max_depth=None,
        max_features='log2',
        max_leaf_nodes=None,
        max_samples=None,
        min_impurity_decrease=0.0,
        min_samples_leaf=0.0028629618034842247,
        min_samples_split=0.0037087774117744725,
        min_weight_fraction_leaf=0.0,
        n_estimators=50,
        n_jobs=-1,
        oob_score=False,
        random_state=None,
        verbose=0,
        warm_start=False
    )
    
    return algorithm
    
    
def generate_preprocessor_config_6():
    from sklearn.preprocessing import MinMaxScaler
    
    preproc = MinMaxScaler(
        clip=False,
        copy=True,
        feature_range=(0, 1)
    )
    
    return preproc
    
    
def generate_algorithm_config_6():
    from sklearn.ensemble import ExtraTreesRegressor
    
    algorithm = ExtraTreesRegressor(
        bootstrap=True,
        ccp_alpha=0.0,
        criterion='mse',
        max_depth=None,
        max_features=0.8,
        max_leaf_nodes=None,
        max_samples=None,
        min_impurity_decrease=0.0,
        min_samples_leaf=0.07388264419939175,
        min_samples_split=0.04427451843494491,
        min_weight_fraction_leaf=0.0,
        n_estimators=25,
        n_jobs=-1,
        oob_score=False,
        random_state=None,
        verbose=0,
        warm_start=False
    )
    
    return algorithm
    
    
def generate_preprocessor_config_7():
    from sklearn.preprocessing import StandardScaler
    
    preproc = StandardScaler(
        copy=True,
        with_mean=True,
        with_std=True
    )
    
    return preproc
    
    
def generate_algorithm_config_7():
    from sklearn.ensemble import ExtraTreesRegressor
    
    algorithm = ExtraTreesRegressor(
        bootstrap=True,
        ccp_alpha=0.0,
        criterion='mse',
        max_depth=None,
        max_features=0.8,
        max_leaf_nodes=None,
        max_samples=None,
        min_impurity_decrease=0.0,
        min_samples_leaf=0.019375063213477914,
        min_samples_split=0.02180025323490051,
        min_weight_fraction_leaf=0.0,
        n_estimators=10,
        n_jobs=-1,
        oob_score=False,
        random_state=None,
        verbose=0,
        warm_start=False
    )
    
    return algorithm
    
    
def generate_preprocessor_config_8():
    from sklearn.preprocessing import RobustScaler
    
    preproc = RobustScaler(
        copy=True,
        quantile_range=[10, 90],
        unit_variance=False,
        with_centering=False,
        with_scaling=False
    )
    
    return preproc
    
    
def generate_algorithm_config_8():
    from sklearn.ensemble import ExtraTreesRegressor
    
    algorithm = ExtraTreesRegressor(
        bootstrap=True,
        ccp_alpha=0.0,
        criterion='mse',
        max_depth=None,
        max_features=0.9,
        max_leaf_nodes=None,
        max_samples=None,
        min_impurity_decrease=0.0,
        min_samples_leaf=0.009017208223328022,
        min_samples_split=0.000753222139758624,
        min_weight_fraction_leaf=0.0,
        n_estimators=10,
        n_jobs=-1,
        oob_score=False,
        random_state=None,
        verbose=0,
        warm_start=False
    )
    
    return algorithm
    
    
def generate_preprocessor_config_9():
    from sklearn.preprocessing import RobustScaler
    
    preproc = RobustScaler(
        copy=True,
        quantile_range=[10, 90],
        unit_variance=False,
        with_centering=True,
        with_scaling=False
    )
    
    return preproc
    
    
def generate_algorithm_config_9():
    from sklearn.linear_model import ElasticNet
    
    algorithm = ElasticNet(
        alpha=0.001,
        copy_X=True,
        fit_intercept=True,
        l1_ratio=0.21842105263157896,
        max_iter=1000,
        normalize=False,
        positive=False,
        precompute=False,
        random_state=None,
        selection='cyclic',
        tol=0.0001,
        warm_start=False
    )
    
    return algorithm
    
    
def generate_algorithm_config():
    '''
    Specifies the actual algorithm and hyperparameters for training the model.
    
    It is the last stage of the final scikit-learn pipeline. For ensemble models, generate_preprocessor_config_N()
    (if needed) and generate_algorithm_config_N() are defined for each learner in the ensemble model,
    where N represents the placement of each learner in the ensemble model's list. For stack ensemble
    models, the meta learner generate_algorithm_config_meta() is defined.
    '''
    from azureml.training.tabular.models.voting_ensemble import PreFittedSoftVotingRegressor
    from sklearn.pipeline import Pipeline
    
    pipeline_0 = Pipeline(steps=[('preproc', generate_preprocessor_config_0()), ('model', generate_algorithm_config_0())])
    pipeline_1 = Pipeline(steps=[('preproc', generate_preprocessor_config_1()), ('model', generate_algorithm_config_1())])
    pipeline_2 = Pipeline(steps=[('preproc', generate_preprocessor_config_2()), ('model', generate_algorithm_config_2())])
    pipeline_3 = Pipeline(steps=[('preproc', generate_preprocessor_config_3()), ('model', generate_algorithm_config_3())])
    pipeline_4 = Pipeline(steps=[('preproc', generate_preprocessor_config_4()), ('model', generate_algorithm_config_4())])
    pipeline_5 = Pipeline(steps=[('preproc', generate_preprocessor_config_5()), ('model', generate_algorithm_config_5())])
    pipeline_6 = Pipeline(steps=[('preproc', generate_preprocessor_config_6()), ('model', generate_algorithm_config_6())])
    pipeline_7 = Pipeline(steps=[('preproc', generate_preprocessor_config_7()), ('model', generate_algorithm_config_7())])
    pipeline_8 = Pipeline(steps=[('preproc', generate_preprocessor_config_8()), ('model', generate_algorithm_config_8())])
    pipeline_9 = Pipeline(steps=[('preproc', generate_preprocessor_config_9()), ('model', generate_algorithm_config_9())])
    algorithm = PreFittedSoftVotingRegressor(
        estimators=[
            ('model_0', pipeline_0),
            ('model_1', pipeline_1),
            ('model_2', pipeline_2),
            ('model_3', pipeline_3),
            ('model_4', pipeline_4),
            ('model_5', pipeline_5),
            ('model_6', pipeline_6),
            ('model_7', pipeline_7),
            ('model_8', pipeline_8),
            ('model_9', pipeline_9),
        ],
        weights=[0.06666666666666667, 0.2, 0.06666666666666667, 0.06666666666666667, 0.06666666666666667, 0.13333333333333333, 0.13333333333333333, 0.06666666666666667, 0.13333333333333333, 0.06666666666666667]
    )
    
    return algorithm
    
    
def build_model_pipeline():
    '''
    Defines the scikit-learn pipeline steps.
    '''
    from sklearn.pipeline import Pipeline
    
    logger.info("Running build_model_pipeline")
    pipeline = Pipeline(
        steps=[
            ('featurization', generate_data_transformation_config()),
            ('ensemble', generate_algorithm_config()),
        ]
    )
    
    return pipeline


def train_model(X, y, sample_weights=None, transformer=None):
    '''
    Calls the fit() method to train the model.
    
    The return value is the model fitted/trained on the input data.
    '''
    
    logger.info("Running train_model")
    model_pipeline = build_model_pipeline()
    
    model = model_pipeline.fit(X, y)
    return model


def calculate_metrics(model, X, y, sample_weights, X_test, y_test, cv_splits=None):
    '''
    Calculates the metrics that can be used to evaluate the model's performance.
    
    Metrics calculated vary depending on the experiment type. Classification, regression and time-series
    forecasting jobs each have their own set of metrics that are calculated.'''
    
    from azureml.training.tabular.preprocessing._dataset_binning import make_dataset_bins
    from azureml.training.tabular.score.scoring import score_regression
    
    y_pred = model.predict(X_test)
    y_min = np.min(y)
    y_max = np.max(y)
    y_std = np.std(y)
    
    bin_info = make_dataset_bins(X_test.shape[0], y_test)
    metrics = score_regression(
        y_test, y_pred, get_metrics_names(), y_max, y_min, y_std, sample_weights, bin_info)
    return metrics


def get_metrics_names():
    
    metrics_names = [
        'explained_variance',
        'predicted_true',
        'normalized_root_mean_squared_log_error',
        'mean_absolute_percentage_error',
        'spearman_correlation',
        'normalized_mean_absolute_error',
        'r2_score',
        'normalized_root_mean_squared_error',
        'root_mean_squared_error',
        'median_absolute_error',
        'normalized_median_absolute_error',
        'mean_absolute_error',
        'root_mean_squared_log_error',
        'residuals',
    ]
    return metrics_names


def get_metrics_log_methods():
    
    metrics_log_methods = {
        'explained_variance': 'log',
        'predicted_true': 'log_predictions',
        'normalized_root_mean_squared_log_error': 'log',
        'mean_absolute_percentage_error': 'log',
        'spearman_correlation': 'log',
        'normalized_mean_absolute_error': 'log',
        'r2_score': 'log',
        'normalized_root_mean_squared_error': 'log',
        'root_mean_squared_error': 'log',
        'median_absolute_error': 'log',
        'normalized_median_absolute_error': 'log',
        'mean_absolute_error': 'log',
        'root_mean_squared_log_error': 'log',
        'residuals': 'log_residuals',
    }
    return metrics_log_methods


def main(training_dataset_uri=None):
    '''
    Runs all functions defined above.
    '''
    
    from azureml.automl.core.inference import inference
    from azureml.core.run import Run
    
    import mlflow
    
    # The following code is for when running this code as part of an AzureML script run.
    run = Run.get_context()
    
    df = get_training_dataset(training_dataset_uri)
    X, y, sample_weights = prepare_data(df)
    split_ratio = 0.25
    (X_train, y_train, sample_weights_train), (X_valid, y_valid, sample_weights_valid) = split_dataset(X, y, sample_weights, split_ratio, should_stratify=False)
    model = train_model(X_train, y_train, sample_weights_train)
    
    metrics = calculate_metrics(model, X, y, sample_weights, X_test=X_valid, y_test=y_valid)
    metrics_log_methods = get_metrics_log_methods()
    print(metrics)
    for metric in metrics:
        if metrics_log_methods[metric] == 'None':
            logger.warning("Unsupported non-scalar metric {}. Will not log.".format(metric))
        elif metrics_log_methods[metric] == 'Skip':
            pass # Forecasting non-scalar metrics and unsupported classification metrics are not logged
        else:
            getattr(run, metrics_log_methods[metric])(metric, metrics[metric])
    cd = inference.get_conda_deps_as_dict(True)
    
    # Saving ML model to outputs/.
    signature = mlflow.models.signature.infer_signature(X, y)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path='outputs/',
        conda_env=cd,
        signature=signature,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE)
    
    run.upload_folder('outputs/', 'outputs/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_dataset_uri', type=str, default='azureml://locations/eastus/workspaces/54535ecf-43bd-4e48-ab3a-f5c71e4dc7c0/data/leopard_dataset_precision/versions/1',     help='Default training dataset uri is populated from the parent run')
    args = parser.parse_args()
    
    try:
        main(args.training_dataset_uri)
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise