import boto3
import json


class SagemakerClient:
    def __init__(self, region_name: str, endpoint_name: str, profile_name: str):
        self.region_name = region_name
        self.endpoint_name = endpoint_name
        self.sagemaker_runtime = boto3.Session(profile_name=profile_name).client(
            "sagemaker-runtime", region_name=region_name
        )

    def invoke_endpoint(self, input_data: str) -> dict:
        response = self.sagemaker_runtime.invoke_endpoint(
            EndpointName=self.endpoint_name, ContentType="text/csv", Body=input_data
        )
        return json.loads(response["Body"].read().decode())

    def predict_prob(self, input_data: str) -> float:
        result = self.invoke_endpoint(input_data)
        return result["probabilities-1d"][0][0]

    def invoke_llama2_endpoint(self, payload):
        response = self.sagemaker_runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload),
            CustomAttributes="accept_eula=true",
        )
        response = response["Body"].read().decode("utf8")
        response = json.loads(response)
        return response


# region = 'us-east-1'
# # Initialize a Boto3 SageMaker client
# # sagemaker = boto3.client('sagemaker', region_name=region)
#
# # Specify the name of your SageMaker endpoint
# endpoint_name = 'jumpstart-sequence-model'
# session = boto3.Session(profile_name='flex-dev')
# sagemaker_runtime = session.client(
#     "sagemaker-runtime",
#     region_name=region)
# # Prepare the input data for inference (replace with your data)
# input_data = "32,10,1,10,8,2,6,41,10000000,8253,63"
# # input_data = pd.DataFrame(
# #         {'sequence_name_cat': [32],
# #          'contact_job_title_level': [10],
# #          'contact_job_title_department': [1],
# #          'company_protfolio_type': [10],
# #          'company_protfolio_subtype': [8],
# #          'company_segment': [2],
# #          'company_state': [6],
# #          'contact_state': [41],
# #          'company_annual_revenue': [10000000],
# #          'company_units': [8253],
# #          'company_founded_years': [63]})
#
# # Make an inference request to the SageMaker endpoint
# response = sagemaker_runtime.invoke_endpoint(
#     EndpointName=endpoint_name,
#     ContentType='text/csv',
#     Body=input_data
# )
# result = json.loads(response['Body'].read().decode())
# print(result)
