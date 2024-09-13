import boto3
import zipfile
import os

def zip_and_upload_to_s3(folder_path, bucket_name, s3_key, aws_access_key_id, aws_secret_access_key):
    # Step 1: Create a ZIP file of the folder
    zip_file_name = f"{folder_path}.zip"
    
    # Zip the entire folder
    with zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zip_file.write(file_path, arcname)

    print(f"Folder {folder_path} zipped successfully as {zip_file_name}")

    # Step 2: Upload the ZIP file to S3 using provided API keys
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
    try:
        s3_client.upload_file(zip_file_name, bucket_name, s3_key)
        print(f"File {zip_file_name} uploaded to S3 as {s3_key}")
    except Exception as e:
        print(f"Error uploading {zip_file_name} to S3: {e}")



