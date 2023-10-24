"""Example workflow pipeline script for abalone pipeline.

                                               . -ModelStep
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""
import os

import boto3
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator

from sagemaker.inputs import ( TrainingInput, TransformInput)

from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.transformer import (
    Transformer
)

from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionGreaterThan
from sagemaker.workflow.condition_step import (
    ConditionStep,
)
from sagemaker.workflow.functions import (
    JsonGet,
    Join
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
    TuningStep,
    TransformStep
)
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.tuner import (
    ContinuousParameter,
    HyperparameterTuner,
    WarmStartConfig,
    WarmStartTypes,
)
from sagemaker.workflow.steps import CacheConfig

cache_config = CacheConfig(enable_caching=True, expire_after="PT1H")


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_sagemaker_client(region):
     """Gets the sagemaker client.

        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts

        Returns:
            `sagemaker.session.Session instance
        """
     boto_session = boto3.Session(region_name=region)
     sagemaker_client = boto_session.client("sagemaker")
     return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_name=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.describe_project(ProjectName=sagemaker_project_name)
        sagemaker_project_arn = response["ProjectArn"]
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline_inference(
    region,
    sagemaker_project_name=None,
    role=None,
    default_bucket=None,
    use_case="ExampleUseCase",
    model_package_group_name="PLXClassificationPackageGroup",
    pipeline_name="PLXClassificationPipelineInference",
    base_job_prefix="PLXClassificationInference",
    model_name="s3_path_example_model.tar.gz",
    processing_instance_type="ml.m5.xlarge",
    inference_instance_type="ml.m5.xlarge",
):
    """Gets a SageMaker ML Pipeline instance working with on abalone data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    pipeline_session = get_pipeline_session(region, default_bucket)

    # parameters for pipeline execution
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)

    inference_instance_type = ParameterString(
    name="InferenceInstanceType",
    default_value=inference_instance_type,
    )

    input_data = ParameterString(
        name="InputDataUrl",
        default_value="example_input_data_s3",
    )
    model_name = ParameterString(
        name='ModelName',default_value="s3_example_model.tar.gz")
    
    # eg GSA for product or ExampleUseCase
    use_case = ParameterString(
        name="UseCase", default_value=use_case
    )

    # e.g TM for telesales
    base_job_prefix = ParameterString(
        name="BaseJobPrefix", default_value=base_job_prefix
    )

    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.0-1",
        py_version="py3",
        instance_type='ml.m5.xlarge'
    )
    
    model = Model(
        image_uri=image_uri,
        model_data=model_name,
        sagemaker_session=pipeline_session,
        role=role,
    )
    
    step_create_model = ModelStep(
    name="PLXClassificationCreateModel",
    step_args=model.create(instance_type="ml.m5.large"),
    )

    

    from datetime import datetime
    now = datetime.now()

    # Convert to string in a specific format
    date_time_string = now.strftime('%Y-%m-%d-%H:%M:%S')

    #Join(on='/',values=['s3:/',sagemaker_session.default_bucket(),f'inference-results-{date_time_string}'])
    
    transformer = Transformer(
    model_name=step_create_model.properties.ModelName,
    instance_type="ml.m5.2xlarge",
    instance_count=1,
    strategy='MultiRecord',
    max_payload=5,
    base_transform_job_name="machine-learning-plx-classification-inference",
    output_path=Join(on='/',values=['s3:/',sagemaker_session.default_bucket(),base_job_prefix,use_case,f'inference-results-{date_time_string}']),
    assemble_with = 'Line', 
    accept = 'text/csv'
    )
    input_location = "example_s3_path" # use input data with ID column

    # Be carefull this needs to be adjusted based on the input data
    step_transform = TransformStep(
        name="PLXClassificationTransform", 
        transformer=transformer, inputs=TransformInput(input_location, split_type='Line',content_type='text/csv', input_filter='$[:-1]', join_source='Input', output_filter='$[-2,-1]')
    )

    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            inference_instance_type,
            model_name,
            input_data,
            use_case
        ],
        steps=[step_create_model, step_transform],
        sagemaker_session=pipeline_session,
    )
    return pipeline
