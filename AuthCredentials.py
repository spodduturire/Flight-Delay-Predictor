import boto3
import pandas

client = boto3.client(
    's3',
    aws_access_key_id = 'XXXXXXXXXXXXXXXXXXXX',
    aws_secret_access_key = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX',
    region_name = 'us-east-1'
)

resource = boto3.resource(
    's3',
    aws_access_key_id = 'XXXXXXXXXXXXXXXXXXXX',
    aws_secret_access_key = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX',
    region_name = 'us-east-1'
)
