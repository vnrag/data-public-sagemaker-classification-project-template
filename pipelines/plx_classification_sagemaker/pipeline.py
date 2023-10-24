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
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
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
    TuningStep
)
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.tuner import (
    ContinuousParameter,
    IntegerParameter,
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


def get_pipeline(
    region,
    sagemaker_project_name=None,
    role=None,
    default_bucket=None,
    model_package_group_name="PLXClassificationPackageGroup",
    pipeline_name="PLXClassificationPipeline",
    base_job_prefix="PLXClassification",
    use_case="EXAMPEMODEL",
    image_uri="492215442770.dkr.ecr.eu-central-1.amazonaws.com/sagemaker-xgboost:1.0-1-cpu-py3",
    processing_instance_type="ml.m5.xlarge",
    training_instance_type="ml.m5.xlarge",
    evaluation_script='evaluate.py'
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

    
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )
    input_data = ParameterString(
        name="InputDataUrl",
        default_value="example_s3_path",
    )
    use_case = ParameterString(
        name="UseCase", default_value=use_case
    )

    base_job_prefix = ParameterString(
        name="BaseJobPrefix", default_value=base_job_prefix
    )

    image_uri = ParameterString(
        name="ImageUri", default_value=image_uri
    )

    evaluation_script = ParameterString(
        name="EvaluationScript", default_value=evaluation_script
    )

    sklearn_processor = SKLearnProcessor(
        role=role,
        framework_version="0.23-1",
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name='machine-learning-plx-classification',
        sagemaker_session=pipeline_session,
    )

    step_args = sklearn_processor.run(
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
        arguments=["--input-data", "example_s3_path"],
        #arguments=["--input-data", input_data],
    )
    step_process = ProcessingStep(
        name="PreprocessTMGSAData",
        step_args=step_args
    )

    # training step for generating model artifacts
    model_path = Join(on='/',values=['s3:/',sagemaker_session.default_bucket(),base_job_prefix,use_case,'train'])
    
    xgb_train = Estimator(
        image_uri=image_uri,
        instance_type=training_instance_type,
        instance_count=1,
        output_path=model_path,
        base_job_name="machine-learning-plx-classification",
        sagemaker_session=pipeline_session,
        role=role,
    )


    xgb_train.set_hyperparameters(
        eval_metric="f1",
        objective="binary:logistic",
        num_round=5,
        max_depth=2,
        eta=0.2,
        gamma=4,
        min_child_weight=6,
        subsample=0.7,
        silent=0,
    )

    objective_metric_name = "validation:f1"

    # Define the hyperparameter ranges
    hyperparameter_ranges = {
        "eta": ContinuousParameter(0.01, 0.3, scaling_type="Linear"),
        "max_depth": IntegerParameter(3, 10),
        "min_child_weight": IntegerParameter(1, 10),
        "subsample": ContinuousParameter(0.7, 1.0, scaling_type="Linear"),
        "gamma": ContinuousParameter(0, 10, scaling_type="Linear"),
        "num_round": IntegerParameter(50, 200, scaling_type="Linear"),
        "alpha": ContinuousParameter(0.001, 1.0, scaling_type="Logarithmic"),
        "lambda": ContinuousParameter(0.001, 1.0, scaling_type="Logarithmic"),
    }
    
    tuner_log = HyperparameterTuner(
        xgb_train,
        objective_metric_name,
        hyperparameter_ranges,
        max_jobs=50,
        max_parallel_jobs=5,
        strategy="Bayesian",
        objective_type="Maximize",
    )

    step_args = tuner_log.fit(
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        }
    )


    step_tuning = TuningStep(
        name="TuningPLXClassificationModel",
        step_args=step_args
    )

    # processing step for evaluation
    script_eval = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name="machine-learning-plx-classification",
        sagemaker_session=pipeline_session,
        role=role,
    )
    step_args = script_eval.run(
        inputs=[
            ProcessingInput(
                source=step_tuning.get_top_model_s3_uri(
                top_k=0, s3_bucket=default_bucket, prefix=Join(on='/',values=[base_job_prefix,use_case,'train'])
            ),
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
        ],
        code=Join=(on='/',values=[BASE_DIR,evaluation_script]),
    )
    evaluation_report = PropertyFile(
        name="PLXClassificationEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    step_eval = ProcessingStep(
        name="EvaluatePLXClassificationModel",
        step_args=step_args,
        property_files=[evaluation_report],
    )

    # register model step that will be conditionally executed
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json"
        )
    )
    model = Model(
        image_uri=image_uri,
        model_data=step_tuning.get_top_model_s3_uri(
        top_k=0, s3_bucket=default_bucket, prefix=Join(on='/',values=[base_job_prefix,use_case,'train'])
    ),
        sagemaker_session=pipeline_session,
        role=role,
    )
    step_args = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics
    )
    step_register = ModelStep(
        name="RegisterPLXClassificationModel",
        step_args=step_args,
    )
    
    # model_name = step_register.properties.ModelName

    # condition step for evaluating model quality and branching execution
    
    # f1 > 0.5
    cond_lte = ConditionGreaterThan(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="regression_metrics.f1"
        ),
        right=0.5,
    )
    step_cond = ConditionStep(
        name="Checkf1PLXClassificationEvaluation",
        conditions=[cond_lte],
        if_steps=[step_register],
        else_steps=[],
    )
    
    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            model_approval_status,
            input_data,
            image_uri,
            use_case,
            base_job_prefix,
            evaluation_script
        ],
        steps=[step_process, step_tuning, step_eval, step_cond],
        sagemaker_session=pipeline_session,
    )
    return pipeline
